"""Standalone harness for b12x's dense block-scaled GEMM at the Flash Next prefill shapes (GB10, sm_121).

IMPORTANT: in the served checkpoint (hf_quant_config.json, quant_algo MIXED_PRECISION) every dense
projection timed here (GDN in_proj_qkv/z/a/b, out_proj, shared_expert gate/up/down, attention q/k/v/o)
is **MXFP8** (E4M3 values, UE8M0 scales per 32). Only routed experts are NVFP4. --recipe mxfp8 is
therefore the live path; --recipe nvfp4 is the counterfactual "same shape with NVFP4 weights".

Paths timed per shape (M = 4096 rows unless --m):
  live     : BF16-input b12x path (api.mm on a packed weight with a caller-provided workspace), as
             vLLM's B12xBlockscaledLinear runs it: BlockscaledQuery capacity regime (num_tokens=4096),
             mode "quantized" -> b12x Triton _quantize + DenseGemmKernel. Split per kernel with
             torch.profiler.
  serial   : (nvfp4 only) vLLM scaled_fp4_quant + api.mm_nvfp4 with the FixedBlockscaledQuery plan.
  sweep    : the dense kernel called directly (b12x._lib.dense_gemm.dense_gemm) with launch overrides
             -- the knob space b12x's DenseGemmQuery tuner searches (gemm/_tuning.py knob_values):
             nvfp4: tile_m{64,128} tile_n{16..128} tile_k{128,256,512} load_path swap_ab occupancy
             mxfp8: tile_m{16..128} tile_n{16..128} tile_k{64,128} swap_ab large_m_unroll
  vendor   : FlashInfer mm_fp4 / mm_mxfp8 and torch._scaled_mm.

Weights are random legal values with random block scales. Correctness: outputs are compared to an FP32
reference built from the dequantized quantized activations and dequantized weights.

Timing: kernel durations from torch.profiler (CUPTI). "cold" = a 96 MB buffer (HARNESS_FLUSH_MB) is rewritten before each
launch so operands come from DRAM; "hot" = back-to-back launches. CUDA-event times are also recorded.

Usage (in the container):
  B12X_COMPILE_CACHE_DIR=/tmp/b12x-gemm-exp python3 dense_gemm_harness.py default --recipe mxfp8
  ... sweep --recipe mxfp8 --shapes out [--configs "128x128:128:0:1,..."]
  ... vendor --recipe mxfp8
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import traceback

import torch

SHAPES = {
    # name: (N, K)
    "qkvz": (16384, 2560),
    "ba": (96, 2560),
    "out": (2560, 6144),
    "gu": (1280, 2560),
    "down": (2560, 640),
    "attn_qkv": (13312, 2560),
}
PEAK_TFLOPS = 498.0
BW_GBS = 225.0
DEV = torch.device("cuda", 0)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return -1.0


def guard_memory():
    avail = mem_available_gib()
    if 0 <= avail < 8.0:
        print(f"ABORT: MemAvailable {avail:.1f} GiB < 8 GiB", flush=True)
        sys.exit(3)
    return avail


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    lut = E2M1.to(packed.device)
    lo = lut[(packed & 0xF).long()]
    hi = lut[(packed >> 4).long()]
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], packed.shape[1] * 2)


def unswizzle_scale(swz: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of b12x intrinsics.swizzle_block_scale for a [rows_pad, cols_pad] tensor."""
    rp, cp = (rows + 127) // 128 * 128, (cols + 3) // 4 * 4
    s = swz.reshape(-1)[: rp * cp].view(torch.float8_e4m3fn).float()
    # swizzled layout: (rp/128, cp/4, 32, 4, 4) <- padded.reshape(rp/128, 4, 32, cp/4, 4).permute(0,3,2,1,4)
    s = s.reshape(rp // 128, cp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rp, cp)
    return s[:rows, :cols]


def dequant(packed, scale_unswz, gscale_mult):
    vals = unpack_fp4(packed)
    return vals * scale_unswz.repeat_interleave(16, dim=1) * gscale_mult


class Problem:
    def __init__(self, name, m, seed=0):
        from b12x._lib import intrinsics
        import b12x.gemm.blockscaled as api
        self.api = api
        self.name = name
        self.m = m
        self.n, self.k = SHAPES[name]
        g = torch.Generator(device=DEV).manual_seed(seed)
        n, k = self.n, self.k
        self.w_packed = torch.randint(0, 256, (n, k // 2), device=DEV, dtype=torch.uint8, generator=g)
        w_sf = (torch.rand((n, k // 16), device=DEV, generator=g) * 0.75 + 0.25).to(torch.float8_e4m3fn)
        self.w_sf_raw = w_sf
        self.w_sf_swz = intrinsics.swizzle_block_scale(w_sf)
        self.w_gs = torch.tensor([0.02], device=DEV, dtype=torch.float32)  # weight multiplier
        self.weight = api.pack_weight(self.w_packed, self.w_sf_swz, recipe="nvfp4", global_scale=self.w_gs)
        self.x = torch.randn((m, k), device=DEV, generator=g, dtype=torch.float32).to(torch.bfloat16)
        amax = self.x.abs().max().float()
        # ModelOpt convention: input_scale = amax / (448*6); quantizer multiplier = 1/input_scale.
        self.a_gs_inv = (448.0 * 6.0 / amax).reshape(1).float()
        self.alpha = (self.w_gs / self.a_gs_inv).reshape(1).float()
        self._wdeq = None

    @property
    def flops(self):
        return 2.0 * self.m * self.n * self.k

    def bytes_floor(self):
        m, n, k = self.m, self.n, self.k
        a = m * k // 2 + m * k // 16
        w = n * k // 2 + n * k // 16
        c = m * n * 2
        return a + w + c

    def wdeq(self):
        if self._wdeq is None:
            self._wdeq = dequant(self.w_packed, self.w_sf_raw.float(), self.w_gs.item())
        return self._wdeq

    def quant_serial(self):
        from vllm._custom_ops import scaled_fp4_quant
        return scaled_fp4_quant(self.x, self.a_gs_inv, is_sf_swizzled_layout=True)

    def ref_from_quant(self, xq, xs):
        a = dequant(xq, unswizzle_scale(xs, self.m, self.k // 16), 1.0 / self.a_gs_inv.item())
        return a @ self.wdeq().t()

    def ref_bf16(self):
        return self.x.float() @ self.wdeq().t()


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


_FLUSH = None


def flush_l2():
    global _FLUSH
    if _FLUSH is None:
        # Must exceed L2: 96 MB was sized for GB10 (24 MB L2); use HARNESS_FLUSH_MB=512 on a 128 MB-L2 part.
        _FLUSH = torch.empty(int(os.environ.get("HARNESS_FLUSH_MB", "96")) * 1024 * 1024 // 4, device=DEV,
                             dtype=torch.float32)
    _FLUSH.fill_(1.0)


def time_fn(fn, iters=20, warmup=3, cold=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if not cold:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / iters, None
    ts = []
    for _ in range(iters):
        flush_l2()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[0]


def kernel_times(fn, iters=10, cold=True):
    """Per-kernel GPU durations (us, median over iters) via torch.profiler."""
    from torch.profiler import profile, ProfilerActivity
    fn(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if cold:
                flush_l2()
            fn()
        torch.cuda.synchronize()
    per = {}
    for ev in prof.events():
        if ev.device_type.name != "CUDA":
            continue
        name = ev.name
        if "fill" in name.lower() and "Kernel" not in name and "_quantize" not in name:
            continue
        if name.startswith("void at::native::vectorized_elementwise_kernel") or "FillFunctor" in name:
            continue
        per.setdefault(name, []).append(ev.device_time if hasattr(ev, "device_time") else ev.cuda_time)
    out = {}
    for name, ts in per.items():
        ts = sorted(ts)
        out[name] = (ts[len(ts) // 2], len(ts))
    return out


def short_kernel(name):
    if "DenseGemm" in name:
        return "DenseGemmKernel"
    return name[:60]


def default_lowering_info(p: Problem, expected_m=None, output_provided=True):
    """The tile/policy b12x's default (untuned) lowering picks for the quantized-mode GEMM."""
    from b12x.gemm._preparation import _default_lowering
    from b12x.gemm._tuning import DenseGemmQuery
    from b12x.preparation import detect_device
    dev = detect_device(DEV)
    q = DenseGemmQuery(
        recipe="nvfp4", entry_point="gemm.blockscaled.mm", weight_storage="native", output_dtype="bfloat16",
        batch=1, max_rows=p.m, in_features=p.k, out_features=p.n,
        output_mode="provided" if output_provided else "functional", alpha_mode="tensor", expected_m=expected_m,
    )
    low = _default_lowering(q, dev.identity)
    from b12x._lib import dense_gemm as dense
    occ = low.target_occupancy_override
    if occ is None:
        occ = dense._dense_gemm_target_occupancy(
            n=low.n, k=low.k, l=low.l, ab_dtype=dense.get_cutlass_dtype(low.ab_dtype),
            c_dtype=dense.get_cutlass_dtype(low.c_dtype), tile_k=low.tile_k, mma_tiler_mn=low.mma_tiler_mn,
            cluster_shape_mn=low.cluster_shape_mn, sm_count=low.sm_count, load_path=low.load_path,
            swap_ab=low.swap_ab, b_tile_major=low.b_tile_major)
    return dict(tile=low.mma_tiler_mn, tile_k=low.tile_k, load_path=low.load_path, swap_ab=low.swap_ab,
                split_k=low.policy.split_k_slices, single_tile_per_cta=low.policy.single_work_tile_per_cta,
                occupancy=occ, sm_count=low.sm_count, cluster=low.cluster_shape_mn, expected_m=low.expected_m)


# ---------------------------------------------------------------- live / serialized paths

def run_default(p: Problem, iters, results):
    api = p.api
    r = dict(shape=p.name, M=p.m, N=p.n, K=p.k, flops=p.flops, bytes_floor=p.bytes_floor())
    r["floor_bytes_us"] = p.bytes_floor() / (BW_GBS * 1e3)
    r["floor_peak_us"] = p.flops / (PEAK_TFLOPS * 1e6)
    r["default_lowering"] = default_lowering_info(p)

    # live BF16-input path (mode resolved by b12x's default config for a capacity-4096 query)
    live = lambda: api.mm(p.x, p.weight, activation_global_scale=p.a_gs_inv)
    out_live = live(); torch.cuda.synchronize()
    from b12x.gemm.blockscaled._preparation import heuristic_plan, _HEURISTIC_PLANS
    for (dev, q), decl in _HEURISTIC_PLANS.items():
        if getattr(q, "out_features", None) == p.n and getattr(q, "in_features", None) == p.k:
            st = getattr(getattr(decl, "_prepared", None), "state", None)
            cfg = getattr(st, "config", None)
            r["live_query"] = dict(num_tokens=q.num_tokens, expected_m=q.expected_m, mode=getattr(cfg, "mode", None))
    r["live_hot_us"], _ = time_fn(live, iters)
    r["live_cold_us"], r["live_cold_min_us"] = time_fn(live, iters, cold=True)
    kt = kernel_times(live, iters=10, cold=True)
    r["live_kernels_cold"] = {short_kernel(k): v[0] for k, v in kt.items()}
    kt = kernel_times(live, iters=10, cold=False)
    r["live_kernels_hot"] = {short_kernel(k): v[0] for k, v in kt.items()}

    # serialized a4 path (vLLM quant + FixedBlockscaledQuery plan)
    plan = api.plan(api.FixedBlockscaledQuery(
        recipe="nvfp4", call_kind="serialized", max_rows=p.m, in_features=p.k, padded_in_features=p.k,
        out_features=p.n, input_dtype="uint8", output_dtype="bfloat16", expected_m=p.m, alpha_mode="tensor"))
    xq, xs = p.quant_serial()
    gemm = lambda: api.mm_nvfp4(xq, xs, p.w_packed, p.w_sf_swz, p.alpha, plan=plan, out_dtype=torch.bfloat16)
    out_ser = gemm(); torch.cuda.synchronize()
    r["serial_gemm_hot_us"], _ = time_fn(gemm, iters)
    r["serial_gemm_cold_us"], r["serial_gemm_cold_min_us"] = time_fn(gemm, iters, cold=True)
    def qg():
        a, b = p.quant_serial()
        return api.mm_nvfp4(a, b, p.w_packed, p.w_sf_swz, p.alpha, plan=plan, out_dtype=torch.bfloat16)
    r["serial_q+gemm_cold_us"], _ = time_fn(qg, iters, cold=True)
    kt = kernel_times(qg, iters=10, cold=True)
    r["serial_kernels_cold"] = {short_kernel(k): v[0] for k, v in kt.items()}

    # correctness
    ref_q = p.ref_from_quant(xq, xs)
    r["err_serial_vs_qref"] = rel_err(out_ser, ref_q)
    r["err_live_vs_qref"] = rel_err(out_live, ref_q)
    r["err_live_vs_serial"] = rel_err(out_live, out_ser.float())
    r["err_live_vs_bf16ref"] = rel_err(out_live, p.ref_bf16())
    g = r["live_kernels_cold"].get("DenseGemmKernel")
    if g:
        r["live_gemm_cold_tflops"] = p.flops / (g * 1e6)
        r["live_gemm_cold_pct_peak"] = 100 * r["live_gemm_cold_tflops"] / PEAK_TFLOPS
    results.append(r)
    print(json.dumps(r, default=str), flush=True)
    return out_ser


# ---------------------------------------------------------------- sweep

def parse_cfg(s):
    tile, tk, lp, swap, occ = s.split(":")
    tm, tn = (int(v) for v in tile.split("x"))
    return dict(tile=(tm, tn), tile_k=int(tk), load_path=lp, swap_ab=bool(int(swap)),
                occupancy=None if occ in ("", "d") else int(occ))


def default_grid(p: Problem):
    cfgs = []
    for tm, tn, tk, lp, swap, occ in itertools.product(
            (64, 128), (32, 64, 128), (128, 256), ("tma",), (False, True), (1, 2)):
        if tn < 64 and not swap:
            continue
        if p.k % tk:
            continue
        cfgs.append(dict(tile=(tm, tn), tile_k=tk, load_path=lp, swap_ab=swap, occupancy=occ))
    return cfgs


def legal(p: Problem, c):
    """b12x's own validate_config (can_implement + smem minimum + K divisibility)."""
    from b12x.gemm._tuning import DenseGemmQuery, DenseGemmConfig, validate_config
    from b12x.preparation import detect_device
    q = DenseGemmQuery(recipe="nvfp4", entry_point="gemm.blockscaled.mm", weight_storage="native",
                       output_dtype="bfloat16", batch=1, max_rows=p.m, in_features=p.k, out_features=p.n,
                       output_mode="provided", alpha_mode="tensor")
    cfg = DenseGemmConfig(backend="cutedsl", tile_m=c["tile"][0], tile_n=c["tile"][1], tile_k=c["tile_k"],
                          load_path=c["load_path"], swap_ab=c["swap_ab"], split_k_slices=None,
                          large_m_unroll=None, target_occupancy=c["occupancy"])
    try:
        validate_config(q, cfg, detect_device(DEV).identity)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)[:120]


def run_sweep(p: Problem, cfgs, iters, results, ref_out=None):
    from b12x._lib.dense_gemm import dense_gemm
    from b12x._lib.intrinsics import as_grouped_scale_view
    from b12x.gemm.blockscaled._a16 import scale_storage
    xq, xs = p.quant_serial()
    m, n, k = p.m, p.n, p.k
    sfa = as_grouped_scale_view(scale_storage(xs, m, k, 16).view(1, -1), m, k)
    sfb = as_grouped_scale_view(scale_storage(p.w_sf_swz, n, k, 16).view(1, -1), n, k)
    a3 = xq.view(m, k // 2, 1)
    b3 = p.w_packed.view(n, k // 2, 1)
    out = torch.empty((m, n, 1), device=DEV, dtype=torch.bfloat16)
    ref_q = p.ref_from_quant(xq, xs)
    base = None
    for c in cfgs:
        guard_memory()
        why = legal(p, c)
        rec = dict(shape=p.name, M=m, N=n, K=k, **{kk: (list(v) if isinstance(v, tuple) else v) for kk, v in c.items()})
        if why is not None:
            rec["skip"] = why
            results.append(rec); print(json.dumps(rec), flush=True)
            continue
        kw = dict(ab_dtype="float4_e2m1fn", sf_dtype="float8_e4m3fn", c_dtype="bfloat16", sf_vec_size=16,
                  alpha=p.alpha, mma_tiler_mn=c["tile"], _tile_k_override=c["tile_k"], load_path=c["load_path"],
                  swap_ab=c["swap_ab"])
        if c["occupancy"] is not None:
            kw["_target_occupancy_override"] = c["occupancy"]
        fn = lambda: dense_gemm((a3, sfa), (b3, sfb), out, **kw)
        t0 = time.time()
        try:
            fn(); torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            results.append(rec); print(json.dumps(rec), flush=True)
            continue
        rec["compile_s"] = round(time.time() - t0, 1)
        rec["err_vs_qref"] = rel_err(out[:, :, 0], ref_q)
        if ref_out is not None:
            rec["err_vs_default"] = rel_err(out[:, :, 0], ref_out.float())
        rec["hot_us"], _ = time_fn(fn, iters)
        rec["cold_us"], rec["cold_min_us"] = time_fn(fn, iters, cold=True)
        kt = kernel_times(fn, iters=max(5, iters // 2), cold=True)
        rec["kern_cold_us"] = sum(v[0] for kk, v in kt.items() if "DenseGemm" in kk or "reduce" in kk.lower())
        kt = kernel_times(fn, iters=max(5, iters // 2), cold=False)
        rec["kern_hot_us"] = sum(v[0] for kk, v in kt.items() if "DenseGemm" in kk or "reduce" in kk.lower())
        rec["tflops_cold"] = p.flops / (rec["kern_cold_us"] * 1e6)
        results.append(rec)
        print(json.dumps(rec), flush=True)
    return results


# ---------------------------------------------------------------- vendor

def run_vendor(p: Problem, iters, results):
    xq, xs = p.quant_serial()
    ref_q = p.ref_from_quant(xq, xs)
    m, n, k = p.m, p.n, p.k
    # FlashInfer mm_fp4: a [m, k/2] fp4x2, b.T [k/2, n], scales swizzled (128x4) e4m3, alpha fp32 scalar.
    try:
        import flashinfer
        from flashinfer import mm_fp4
        for backend in ("cutlass", "cudnn", "cute-dsl", "trtllm", "b12x"):
            rec = dict(shape=p.name, M=m, N=n, K=k, vendor=f"flashinfer.mm_fp4[{backend}]", flashinfer=flashinfer.__version__)
            try:
                a_sf = xs.view(torch.float8_e4m3fn)
                b_sf = p.w_sf_swz.view(torch.float8_e4m3fn)
                fn = lambda: mm_fp4(xq, p.w_packed.T, a_sf, b_sf.T, p.alpha, torch.bfloat16, backend=backend)
                o = fn(); torch.cuda.synchronize()
                rec["err_vs_qref"] = rel_err(o, ref_q)
                rec["hot_us"], _ = time_fn(fn, iters)
                rec["cold_us"], rec["cold_min_us"] = time_fn(fn, iters, cold=True)
                rec["tflops_cold"] = p.flops / (rec["cold_us"] * 1e6)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            results.append(rec); print(json.dumps(rec), flush=True)
    except Exception as exc:  # noqa: BLE001
        rec = dict(shape=p.name, vendor="flashinfer", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        results.append(rec); print(json.dumps(rec), flush=True)
    # torch._scaled_mm with NVFP4 (float4_e2m1fn_x2 + e4m3 blockwise 1x16 scales, swizzled)
    rec = dict(shape=p.name, M=m, N=n, K=k, vendor="torch._scaled_mm[nvfp4]", torch=torch.__version__)
    try:
        a = xq.view(torch.float4_e2m1fn_x2)
        b = p.w_packed.view(torch.float4_e2m1fn_x2)
        sa = xs.view(torch.float8_e4m3fn).reshape(-1)
        sb = p.w_sf_swz.view(torch.float8_e4m3fn).reshape(-1)
        fn = lambda: torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)
        o = fn(); torch.cuda.synchronize()
        o = o.float() * p.alpha.item()
        rec["err_vs_qref"] = rel_err(o, ref_q)
        rec["hot_us"], _ = time_fn(fn, iters)
        rec["cold_us"], rec["cold_min_us"] = time_fn(fn, iters, cold=True)
        rec["tflops_cold"] = p.flops / (rec["cold_us"] * 1e6)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    results.append(rec); print(json.dumps(rec), flush=True)
    # cuBLAS BF16 and FP8 for scale
    xb = p.x
    wb = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
    rec = dict(shape=p.name, M=m, N=n, K=k, vendor="cublas.bf16 (reference)")
    fn = lambda: torch.matmul(xb, wb.t())
    rec["hot_us"], _ = time_fn(fn, iters)
    rec["cold_us"], rec["cold_min_us"] = time_fn(fn, iters, cold=True)
    rec["tflops_cold"] = p.flops / (rec["cold_us"] * 1e6)
    results.append(rec); print(json.dumps(rec), flush=True)
    del wb



# ================================================================ MXFP8 (the live recipe)

PEAK8_TFLOPS = 249.0


class Problem8:
    def __init__(self, name, m, seed=0):
        import b12x.gemm.blockscaled as api
        self.api = api
        self.name, self.m = name, m
        self.n, self.k = SHAPES[name]
        n, k = self.n, self.k
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.w_fp8 = (torch.randn((n, k), device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
        self.w_sf = torch.randint(118, 128, (n, k // 32), device=DEV, dtype=torch.uint8, generator=g)
        self.weight = api.pack_weight(self.w_fp8, self.w_sf)
        self.x = torch.randn((m, k), device=DEV, generator=g, dtype=torch.float32).to(torch.bfloat16)
        self._wdeq = None

    flops = Problem.flops

    def bytes_floor(self):
        m, n, k = self.m, self.n, self.k
        return m * k + m * k // 32 + n * k + n * k // 32 + m * n * 2

    def wdeq(self):
        if self._wdeq is None:
            self._wdeq = self.w_fp8.float() * torch.exp2(self.w_sf.float() - 127).repeat_interleave(32, dim=1)
        return self._wdeq

    def quant(self):
        """Same rule as b12x _quantize (GROUP=32, non-FP4 branch): e = ceil(log2(amax/448))."""
        m, k = self.m, self.k
        xb = self.x.float().view(m, k // 32, 32)
        amax = xb.abs().amax(-1)
        ratio = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
        e = torch.clamp(torch.ceil(torch.log2(ratio)), -127, 127)
        xq = (xb / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).view(m, k)
        xs = (e + 127).to(torch.uint8)
        return xq, xs

    def ref_from_quant(self, xq, xs):
        a = xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, dim=1)
        return a @ self.wdeq().t()

    def ref_bf16(self):
        return self.x.float() @ self.wdeq().t()

    def workspace(self):
        from b12x.gemm.blockscaled._a16 import _layout
        return torch.empty(_layout(self.m, self.n, self.k, False, (64, 64, 1))[-1], device=DEV, dtype=torch.uint8)


def dense_query8(p, expected_m=4096):
    from b12x.gemm._tuning import DenseGemmQuery
    return DenseGemmQuery(recipe="mxfp8", entry_point="gemm.blockscaled.mm", weight_storage="native",
                          output_dtype="bfloat16", batch=1, max_rows=p.m, in_features=p.k, out_features=p.n,
                          output_mode="provided", alpha_mode="tensor", expected_m=expected_m)


def lowering_info8(p, expected_m=4096):
    """Default lowering for the live quantized-mode GEMM (_lower_dense_with_hint with the prefill hint)."""
    from b12x.gemm._preparation import _default_lowering
    from b12x.preparation import detect_device
    low = _default_lowering(dense_query8(p, expected_m), detect_device(DEV).identity)
    return dict(tile=low.mma_tiler_mn, tile_k=low.tile_k, load_path=low.load_path, swap_ab=low.swap_ab,
                split_k=low.policy.split_k_slices, large_m_unroll=low.policy.large_m_unroll,
                single_tile_per_cta=low.policy.single_work_tile_per_cta, sm_count=low.sm_count,
                expected_m=low.expected_m, sfb_k_reuse=low.sfb_k_reuse)


def run_default8(p: Problem8, iters, results):
    api = p.api
    r = dict(recipe="mxfp8", shape=p.name, M=p.m, N=p.n, K=p.k, flops=p.flops, bytes_floor=p.bytes_floor())
    r["floor_bytes_us"] = p.bytes_floor() / (BW_GBS * 1e3)
    r["floor_peak_us"] = p.flops / (PEAK8_TFLOPS * 1e6)
    r["default_lowering"] = lowering_info8(p)
    ws = p.workspace()
    live = lambda: api.mm(p.x, p.weight, workspace=ws)
    out_live = live(); torch.cuda.synchronize()
    from b12x.gemm.blockscaled._preparation import _HEURISTIC_PLANS
    for (dev, q), decl in _HEURISTIC_PLANS.items():
        if getattr(q, "out_features", None) == p.n and getattr(q, "in_features", None) == p.k and q.recipe == "mxfp8":
            st = getattr(getattr(decl, "_prepared", None), "state", None)
            cfg = getattr(st, "config", None)
            dense = getattr(st, "dense", None)
            low = getattr(dense, "lowering", None)
            r["live_query"] = dict(num_tokens=q.num_tokens, expected_m=q.expected_m, workspace_form=q.workspace_form,
                                   mode=getattr(cfg, "mode", None),
                                   tile=getattr(low, "mma_tiler_mn", None), tile_k=getattr(low, "tile_k", None),
                                   lowering_expected_m=getattr(low, "expected_m", None))
    r["live_hot_us_events"], _ = time_fn(live, iters)
    r["live_cold_us_events"], _ = time_fn(live, iters, cold=True)
    r["live_kernels_cold"] = {short_kernel(k): v[0] for k, v in kernel_times(live, 10, cold=True).items()}
    r["live_kernels_hot"] = {short_kernel(k): v[0] for k, v in kernel_times(live, 10, cold=False).items()}
    xq, xs = p.quant()
    ref_q = p.ref_from_quant(xq, xs)
    r["err_live_vs_qref"] = rel_err(out_live, ref_q)
    r["err_live_vs_bf16ref"] = rel_err(out_live, p.ref_bf16())
    g = r["live_kernels_cold"].get("DenseGemmKernel")
    if g:
        r["live_gemm_cold_tflops"] = p.flops / (g * 1e6)
        r["live_gemm_cold_pct_peak8"] = 100 * r["live_gemm_cold_tflops"] / PEAK8_TFLOPS
    results.append(r)
    print(json.dumps(r, default=str), flush=True)
    return out_live


def parse_cfg8(s):
    tile, tk, swap, lmu = s.split(":")
    tm, tn = (int(v) for v in tile.split("x"))
    return dict(tile=(tm, tn), tile_k=int(tk), swap_ab=bool(int(swap)),
                large_m_unroll=None if lmu in ("", "d") else bool(int(lmu)))


def default_grid8(p):
    cfgs = []
    for tm, tn, tk, swap, lmu in itertools.product((32, 64, 128), (32, 64, 128), (64, 128), (False, True), (False, True)):
        if p.k % tk:
            continue
        if tk == 64 and (tm != 128 or swap):
            continue  # MXFP8 BK64 requires an unswapped 128-row tile (_validate_mxfp8_bk64_plan)
        cfgs.append(dict(tile=(tm, tn), tile_k=tk, swap_ab=swap, large_m_unroll=lmu))
    return cfgs


def legal8(p, c):
    from b12x.gemm._tuning import DenseGemmConfig, validate_config
    from b12x.preparation import detect_device
    cfg = DenseGemmConfig(backend="cutedsl", tile_m=c["tile"][0], tile_n=c["tile"][1], tile_k=c["tile_k"],
                          load_path="tma", swap_ab=c["swap_ab"], split_k_slices=1,
                          large_m_unroll=c["large_m_unroll"] if c["large_m_unroll"] is not None else True,
                          target_occupancy=None)
    try:
        validate_config(dense_query8(p), cfg, detect_device(DEV).identity)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)[:120]


def make_placements8(p, xq, a_mma, count, seed=1):
    """Fresh copies of A and W at shifted addresses. GEMM time on GB10 depends strongly on where A and W
    land (measured: same kernel/data, 1.25-2.05 ms for out_proj across placements), so every config is
    timed on the same set of placements and summarized across them."""
    g = torch.Generator().manual_seed(seed)
    w = p.weight.weight
    m, n, k = p.m, p.n, p.k
    pads, pl = [], []
    for i in range(count):
        pads.append(torch.empty(int(torch.randint(1, 64, (1,), generator=g)) * (1 << 20) + 4096 * i,
                                dtype=torch.uint8, device=DEV))
        b = w.values.clone()
        pads.append(torch.empty(int(torch.randint(1, 64, (1,), generator=g)) * (1 << 20), dtype=torch.uint8, device=DEV))
        a = xq.clone()
        pl.append(((a.view(m, k, 1), a_mma), (b.view(n, k, 1), w.scale_mma.view(torch.float8_e8m0fnu))))
    return pl, pads


def time_placements(run, placements, reps):
    per = []
    for a, b in placements:
        run(a, b)
    torch.cuda.synchronize()
    raw = [[] for _ in placements]
    for _ in range(reps):
        for i, (a, b) in enumerate(placements):
            flush_l2()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); run(a, b); e.record(); torch.cuda.synchronize()
            raw[i].append(s.elapsed_time(e) * 1000)
    for ts in raw:
        ts.sort(); per.append(ts[len(ts) // 2])
    srt = sorted(per)
    return dict(med_us=round(srt[len(srt) // 2], 1), min_us=round(srt[0], 1), max_us=round(srt[-1], 1),
                per_placement_us=[round(t) for t in per])


def run_sweep8(p: Problem8, cfgs, iters, results, ref_live=None, placements=6):
    from b12x._lib.dense_gemm import dense_gemm
    from b12x.gemm.blockscaled._linear import _mxfp8_scale_mma_from_input
    m, n, k = p.m, p.n, p.k
    xq, xs = p.quant()
    a_mma = _mxfp8_scale_mma_from_input(xs, rows=m, width=k, logical_width=k).view(torch.float8_e8m0fnu)
    pl, pads = make_placements8(p, xq, a_mma, placements)
    out = torch.empty((m, n, 1), device=DEV, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=DEV, dtype=torch.float32)
    ref_q = p.ref_from_quant(xq, xs)
    for c in cfgs:
        guard_memory()
        rec = dict(recipe="mxfp8", shape=p.name, M=m, N=n, K=k, sched=os.environ.get("HARNESS_SCHED", "stock"),
                   **{kk: (list(v) if isinstance(v, tuple) else v) for kk, v in c.items()})
        if c.get("default"):
            kw = dict(expected_m=4096)
        else:
            why = legal8(p, c)
            if why is not None:
                rec["skip"] = why
                results.append(rec); print(json.dumps(rec), flush=True)
                continue
            kw = dict(expected_m=4096, mma_tiler_mn=c["tile"], _tile_k_override=c["tile_k"], swap_ab=c["swap_ab"],
                      load_path="tma")
            if c["large_m_unroll"] is not None:
                kw["_large_m_unroll_override"] = c["large_m_unroll"]
        run = lambda a, b: dense_gemm(a, b, out, ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
                                      c_dtype="bfloat16", sf_vec_size=32, alpha=alpha, **kw)
        t0 = time.time()
        try:
            run(*pl[0]); torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            results.append(rec); print(json.dumps(rec), flush=True)
            continue
        rec["compile_s"] = round(time.time() - t0, 1)
        rec["err_vs_qref"] = rel_err(out[:, :, 0], ref_q)
        if ref_live is not None:
            rec["err_vs_live"] = rel_err(out[:, :, 0], ref_live.float())
        rec.update(time_placements(run, pl, iters))
        kt = kernel_times(lambda: run(*pl[0]), iters=5, cold=True)
        rec["kern_cold_p0_us"] = round(sum(v[0] for kk, v in kt.items() if "DenseGemm" in kk), 1)
        rec["tflops_med"] = round(p.flops / (rec["med_us"] * 1e6), 1)
        results.append(rec)
        print(json.dumps(rec), flush=True)
    del pads
    return results


def run_vendor8(p: Problem8, iters, results):
    m, n, k = p.m, p.n, p.k
    xq, xs = p.quant()
    ref_q = p.ref_from_quant(xq, xs)
    def finish(rec, fn, post=lambda o: o):
        try:
            o = fn(); torch.cuda.synchronize()
            rec["err_vs_qref"] = rel_err(post(o), ref_q)
            kt = kernel_times(fn, iters=10, cold=True)
            rec["kernels_cold"] = {kk[:70]: v[0] for kk, v in kt.items()}
            rec["kern_cold_us"] = sum(v[0] for v in kt.values())
            rec["tflops_cold"] = p.flops / (rec["kern_cold_us"] * 1e6)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        results.append(rec); print(json.dumps(rec, default=str), flush=True)
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale
        import flashinfer
        a_sw = swizzle_mxfp8_scale(xs, M=m, K=k).reshape(-1)
        b_sw = swizzle_mxfp8_scale(p.w_sf, M=n, K=k).reshape(-1)
        for backend in ("cutlass", "cudnn", "cute-dsl", "trtllm", "b12x", "auto"):
            rec = dict(recipe="mxfp8", shape=p.name, M=m, N=n, K=k, vendor=f"flashinfer.mm_mxfp8[{backend}]",
                       flashinfer=flashinfer.__version__)
            finish(rec, lambda: flashinfer.mm_mxfp8(xq, p.w_fp8.t(), a_sw, b_sw, out_dtype=torch.bfloat16, backend=backend))
    except Exception as exc:  # noqa: BLE001
        rec = dict(recipe="mxfp8", shape=p.name, vendor="flashinfer.mm_mxfp8", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        results.append(rec); print(json.dumps(rec), flush=True)
    # torch._scaled_mm MX: e8m0 scales, 1x32 blocks, swizzled 128x4
    rec = dict(recipe="mxfp8", shape=p.name, M=m, N=n, K=k, vendor="torch._scaled_mm[mxfp8]", torch=torch.__version__)
    try:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale
        sa = swizzle_mxfp8_scale(xs, M=m, K=k).view(torch.float8_e8m0fnu).reshape(-1)
        sb = swizzle_mxfp8_scale(p.w_sf, M=n, K=k).view(torch.float8_e8m0fnu).reshape(-1)
        finish(rec, lambda: torch._scaled_mm(xq, p.w_fp8.t(), sa, sb, out_dtype=torch.bfloat16))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        results.append(rec); print(json.dumps(rec), flush=True)
    # per-tensor FP8 cuBLASLt (no block scales): an upper-bound reference for FP8 tensor-core GEMM here
    rec = dict(recipe="mxfp8", shape=p.name, M=m, N=n, K=k, vendor="torch._scaled_mm[fp8 per-tensor, reference]")
    one = torch.ones((), device=DEV, dtype=torch.float32)
    finish(rec, lambda: torch._scaled_mm(xq, p.w_fp8.t(), one, one, out_dtype=torch.bfloat16),
           post=lambda o: o.float())
    rec.pop("err_vs_qref", None)
    rec = dict(recipe="mxfp8", shape=p.name, M=m, N=n, K=k, vendor="cublas.bf16 (reference)")
    wb = torch.randn((n, k), device=DEV, dtype=torch.bfloat16)
    finish(rec, lambda: torch.matmul(p.x, wb.t()))
    del wb



def install_sched_patch():
    """EXPERIMENT ONLY (process-local monkeypatch, no installed file is touched): override the persistent
    tile scheduler's rasterization in b12x DenseGemmKernel._compute_grid.
    HARNESS_SCHED="rm=0,sw=1" -> raster along N, no swizzle; "rm=1,sw=8" -> along M with 8-wide swizzle.
    Must be combined with B12X_COMPILE_DISK_CACHE=0 so no patched kernel is cached under a stock key."""
    spec = os.environ.get("HARNESS_SCHED")
    if not spec:
        return None
    if os.environ.get("B12X_COMPILE_DISK_CACHE", "1") != "0":
        raise SystemExit("HARNESS_SCHED requires B12X_COMPILE_DISK_CACHE=0")
    kv = dict(item.split("=") for item in spec.split(","))
    raster_m, swz = bool(int(kv.get("rm", "1"))), int(kv.get("sw", "1"))
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from b12x._lib import dense_gemm as dense
    orig = dense.DenseGemmKernel._compute_grid

    def _compute_grid(c, tile_shape_mnk, max_active_clusters, direct_one_m_tile_scheduler, split_k_slices,
                      large_m_unroll, split_k_all_m=False):
        if split_k_slices > 1 or split_k_all_m or direct_one_m_tile_scheduler:
            return orig(c, tile_shape_mnk, max_active_clusters, direct_one_m_tile_scheduler, split_k_slices,
                        large_m_unroll, split_k_all_m)
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        params = utils.PersistentTileSchedulerParams(num_ctas_mnl, (1, 1, 1), swizzle_size=swz,
                                                     raster_along_m=raster_m)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(params, max_active_clusters)
        return params, grid

    dense.DenseGemmKernel._compute_grid = staticmethod(_compute_grid)
    return dict(raster_along_m=raster_m, swizzle=swz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("default", "sweep", "vendor", "one"))
    ap.add_argument("--shapes", default="qkvz,ba,out,gu,down")
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--configs", default="", help="comma list tile:tk:load:swap:occ e.g. 128x128:128:tma:0:1")
    ap.add_argument("--json", default="")
    ap.add_argument("--recipe", default="mxfp8", choices=("mxfp8", "nvfp4"))
    ap.add_argument("--placements", type=int, default=6)
    args = ap.parse_args()
    print(f"# MemAvailable {guard_memory():.1f} GiB  torch {torch.__version__}  "
          f"cache={os.environ.get('B12X_COMPILE_CACHE_DIR')} disk={os.environ.get('B12X_COMPILE_DISK_CACHE')}", flush=True)
    sched = install_sched_patch()
    if sched:
        print(f"# SCHEDULER PATCH ACTIVE: {sched}", flush=True)
    results = []
    for name in args.shapes.split(","):
        guard_memory()
        try:
            if args.recipe == "mxfp8":
                p = Problem8(name, args.m)
                if args.mode == "default":
                    run_default8(p, args.iters, results)
                elif args.mode in ("sweep", "one"):
                    cfgs = [parse_cfg8(s) if s != "default" else dict(default=True) for s in args.configs.split(",")] \
                        if args.configs else [dict(default=True)] + default_grid8(p)
                    ref = None
                    if args.mode == "sweep":
                        ws = p.workspace()
                        ref = p.api.mm(p.x, p.weight, workspace=ws)
                    run_sweep8(p, cfgs, args.iters if args.mode == "sweep" else 1, results, ref_live=ref,
                               placements=args.placements if args.mode == "sweep" else 1)
                elif args.mode == "vendor":
                    run_vendor8(p, args.iters, results)
            else:
                p = Problem(name, args.m)
                if args.mode == "default":
                    run_default(p, args.iters, results)
                elif args.mode == "sweep":
                    cfgs = [parse_cfg(s) for s in args.configs.split(",")] if args.configs else default_grid(p)
                    run_sweep(p, cfgs, args.iters, results)
                elif args.mode == "vendor":
                    run_vendor(p, args.iters, results)
                elif args.mode == "one":
                    run_sweep(p, [parse_cfg(s) for s in args.configs.split(",")], 1, results)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        p = None
        torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()

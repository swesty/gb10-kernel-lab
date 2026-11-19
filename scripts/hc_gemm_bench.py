"""Microbenchmark the HyperConnection low-rank projections at decode shapes on GB10.

Shapes (Qwen3.8 Flash Next, hc_count=4, hidden=2560, hc_lowrank=320):
  down: x[M, 10240] @ W[336, 10240]^T  (320 low-rank + 4 injection + 12 padding)
  up:   x[M, 320]   @ W[10240, 320]^T
Weights are cycled across NCOPY distinct copies (> 24 MB L2) so every call streams from DRAM, as in serving.

Implementations: cuBLAS via F.linear (the live path), b12x small-N BF16 GEMV (down only, M <= 8),
FP8 E4M3 weights + activations via torch._scaled_mm (per-tensor scales; activation quantisation timed separately).
Usage: python hc_gemm_bench.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
NCOPY, ITERS = 8, 200
SHAPES = {"down": (10240, 336), "up": (320, 10240)}  # (K, N)
MS = (1, 5, 8, 20, 40)


def bench(fn, iters=ITERS):
    for _ in range(10): fn(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters): fn(i % NCOPY)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters  # us


def b12x_gemv(weight_list, m):
    from b12x.gemm.bf16_gemv import GemvQuery, plan as plan_query, mm
    from b12x.preparation import PreparedCall
    from b12x.preparation.session import prepare_default
    w = weight_list[0]
    query = GemvQuery(source_dtype="bfloat16", weight_dtype="bfloat16", max_rows=m,
                      in_features=w.shape[1], out_features=w.shape[0], source_contiguous=True,
                      source_aligned=True, weight_contiguous=True, weight_aligned=True)
    p = plan_query(query)
    src = torch.randn(m, w.shape[1], dtype=torch.bfloat16, device=dev)
    req = p.request(name=f"bench-gemv-m{m}", prepare_call=lambda state: PreparedCall(
        run=lambda: state.run(src, w), owners=(src,)))
    prepare_default(req)
    return lambda x, i: mm(x, weight_list[i], plan=p)


print(f"{'shape':5s} {'M':>3s} {'impl':28s} {'us':>8s} {'GB/s(weights)':>14s}  note")
for name, (K, N) in SHAPES.items():
    weights = [torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(NCOPY)]
    wbytes_bf16 = N * K * 2
    w8 = [w.to(torch.float8_e4m3fn) for w in weights]
    one = torch.ones((), dtype=torch.float32, device=dev)
    for m in MS:
        x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)
        ref = F.linear(x, weights[0])
        rows = []
        t = bench(lambda i: F.linear(x, weights[i]))
        rows.append(("cuBLAS BF16 (live path)", t, wbytes_bf16, ""))
        if name == "down" and m <= 8:
            try:
                g = b12x_gemv(weights, m)
                out = g(x, 0)
                err = (out.float() - ref.float()).abs().max().item()
                t = bench(lambda i: g(x, i))
                rows.append(("b12x small-N BF16 GEMV", t, wbytes_bf16, f"max err {err:.3g}"))
            except Exception as e:
                rows.append(("b12x small-N BF16 GEMV", float("nan"), wbytes_bf16, f"failed: {type(e).__name__}: {str(e)[:80]}"))
        try:
            x8 = x.to(torch.float8_e4m3fn)
            mpad = max(16, (m + 15) // 16 * 16)  # _scaled_mm wants M multiple of 16 on some builds
            x8p = torch.zeros(mpad, K, dtype=torch.float8_e4m3fn, device=dev); x8p[:m] = x8
            f = lambda i: torch._scaled_mm(x8p, w8[i].t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
            out = f(0)[:m]
            err = (out.float() - ref.float()).abs().max().item()
            t = bench(f)
            tq = bench(lambda i: x.to(torch.float8_e4m3fn))
            rows.append(("FP8 _scaled_mm (W+A e4m3)", t, N * K, f"+{tq:.1f} us act quant; max err {err:.3g}"))
        except Exception as e:
            rows.append(("FP8 _scaled_mm (W+A e4m3)", float("nan"), N * K, f"failed: {type(e).__name__}: {str(e)[:80]}"))
        for impl, t, wb, note in rows:
            print(f"{name:5s} {m:3d} {impl:28s} {t:8.1f} {wb / t / 1e3 if t == t else float('nan'):14.0f}  {note}")

"""Harness for b12x's MTP feedback prefill projection (MTPPrefillBf16GemmKernel) at Flash Next shapes.

The hidden projection is [tokens*4, 2560] x [2560, 2560]^T with a broadcast add of the token path
(one row per token, shared by its 4 streams). In serving it runs once per 4,096-token prefill chunk.
Variants come from an experimental copy of the module (_cute_prefill_exp.py, made by mtp_exp_patch.py)
whose tile N, stage count and rasterisation are read from B12X_MTP_TILE_N / B12X_MTP_STAGES /
B12X_MTP_RASTER_N. Output is checked against a PyTorch reference; cuBLAS BF16 matmul is timed as a
reference for what this GPU reaches on the same GEMM.

Usage: B12X_COMPILE_DISK_CACHE=0 python3 mtp_gemm_harness.py [--exp] [--tokens 4096] [--flush-mb 512]
"""
import argparse
import importlib
import os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stress", type=int, default=0, help="extra launches compared bitwise to the first output")
ap.add_argument("--tokens", type=int, default=4096)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--flush-mb", type=int, default=0,
                help="rewrite a buffer this size before each timed launch so A/W start outside L2 (0 = back-to-back)")
args = ap.parse_args()

mod = importlib.import_module("b12x.sequence.mtp_feedback." + ("_cute_prefill_fence"))
dev = torch.device("cuda", 0)
S, H, T = 4, 2560, args.tokens
rows = T * S
g = torch.Generator(device=dev).manual_seed(0)
x = (torch.randn(rows, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
w = (torch.randn(H, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
tok = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
out = torch.empty(rows * H, device=dev, dtype=torch.bfloat16)

launch = mod.compile_mtp_prefill_bf16_gemm(rows, H, H, device=dev, streams=S, add_token_path=True)


flush = torch.empty(args.flush_mb * 1024 * 1024 // 4, device=dev, dtype=torch.float32) if args.flush_mb else None


def timed(fn, iters):
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if flush is None:
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    ts = []
    for _ in range(iters):  # median of individually timed launches, each after an L2 flush
        flush.fill_(1.0)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2]


ms = timed(lambda: launch(x, w, out, token_path=tok.view(-1), live_rows=rows), args.iters)
ref = ((x @ w.t()).float() + tok.repeat_interleave(S, 0).float())  # BF16 GEMM, then add in FP32
ref_b = ((x @ w.t()).to(torch.bfloat16).float() + tok.repeat_interleave(S, 0).float()).to(torch.bfloat16)
o = out.view(rows, H)
err = ((o.float() - ref).norm() / ref.norm()).item()
exact = (o == ref_b).float().mean().item()
flop = 2.0 * rows * H * H
cfg = {k: os.environ.get(k, "default") for k in ("B12X_MTP_TILE_N", "B12X_MTP_STAGES", "B12X_MTP_RASTER_N", "B12X_MTP_FENCE")}
print(f"{cfg}: {ms * 1000:9.0f} us  {flop / ms / 1e9:6.1f} TFLOPS  rel err {err:.2e}  bit-equal to BF16 ref {100 * exact:.2f}%")
cub = timed(lambda: torch.matmul(x, w.t()), args.iters)
print(f"cuBLAS BF16 matmul (no epilogue): {cub * 1000:9.0f} us  {flop / cub / 1e9:6.1f} TFLOPS")
if args.stress:
    first = out.clone(); bad = 0
    for i in range(args.stress):
        out.zero_()
        launch(x, w, out, token_path=tok.view(-1), live_rows=rows)
        torch.cuda.synchronize()
        if not torch.equal(out, first):
            bad += 1
    print(f"stress: {bad} of {args.stress} launches differ from the first output")

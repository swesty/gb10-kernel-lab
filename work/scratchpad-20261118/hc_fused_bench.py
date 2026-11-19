"""Fused HyperConnection "up projection + gated mean" vs the live two-kernel path (GB10).

Live path (per HC mixer, Qwen3.8 Flash Next: streams=4, hidden=2560, lowrank=320):
  gate_logits = F.linear(bottleneck[T,320], W_up[10240,320])        # cuBLAS, writes T x 10240 BF16
  out[t,h]    = mean_s bf16(bf16(sigmoid(gate_logits[t,s,h])) * normalized[t,s,h])   # b12x Triton _gate_mean
Fused: one Triton kernel computes the 4 per-stream dots over K=320 in registers and applies the gated mean in
the epilogue, so gate_logits never reach memory. Rounding mirrors the live path (logits rounded to BF16,
gate and product rounded to BF16, FP32 accumulation, /streams).
Usage: python hc_fused_bench.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

S, H, K = 4, 2560, 320
dev = "cuda"
NCOPY = 8


@triton.jit
def _gate_mean_kernel(normalized_ptr, gate_logits_ptr, out_ptr, HIDDEN_SIZE: tl.constexpr, STREAMS: tl.constexpr,
                      BLOCK_H: tl.constexpr):
    # Copied from b12x/norm/hyperconnection/_kernels.py (the live kernel), for an exact baseline.
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1).to(tl.int64)
    cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    token_base = token * STREAMS * HIDDEN_SIZE
    total = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(0, STREAMS):
        offsets = token_base + stream * HIDDEN_SIZE + cols
        values = tl.load(normalized_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
        logits = tl.load(gate_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits).to(tl.bfloat16)
        product = (gate * values).to(tl.bfloat16)
        total += product.to(tl.float32)
    out_offsets = token * HIDDEN_SIZE + cols
    tl.store(out_ptr + out_offsets, total / STREAMS, mask=mask)


@triton.jit
def _fused_up_gated_mean_kernel(x_ptr, w_ptr, norm_ptr, out_ptr, T, H: tl.constexpr, K: tl.constexpr,
                                STREAMS: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
                                BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    tmask = t < T
    hmask = h < H
    t64 = t.to(tl.int64)
    total = tl.zeros((BLOCK_T, BLOCK_H), tl.float32)
    for s in tl.static_range(STREAMS):
        acc = tl.zeros((BLOCK_T, BLOCK_H), tl.float32)
        for k0 in tl.static_range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(x_ptr + t64[:, None] * K + k[None, :], mask=tmask[:, None], other=0.0)
            w = tl.load(w_ptr + (s * H + h)[:, None].to(tl.int64) * K + k[None, :], mask=hmask[:, None], other=0.0)
            acc += tl.dot(a, tl.trans(w))
        logits = acc.to(tl.bfloat16).to(tl.float32)
        gate = tl.sigmoid(logits).to(tl.bfloat16)
        vals = tl.load(norm_ptr + t64[:, None] * (STREAMS * H) + s * H + h[None, :],
                       mask=tmask[:, None] & hmask[None, :], other=0.0)
        product = (gate * vals).to(tl.bfloat16)
        total += product.to(tl.float32)
    tl.store(out_ptr + t64[:, None] * H + h[None, :], (total / STREAMS).to(tl.bfloat16),
             mask=tmask[:, None] & hmask[None, :])


def unfused(x, w, normalized, logits, out):
    torch.mm(x, w.t(), out=logits)
    _gate_mean_kernel[(x.shape[0], 1)](normalized, logits, out, HIDDEN_SIZE=H, STREAMS=S, BLOCK_H=4096, num_warps=4)


def fused(x, w, normalized, out, bt, bh, bk, warps, stages):
    T = x.shape[0]
    _fused_up_gated_mean_kernel[(triton.cdiv(T, bt), triton.cdiv(H, bh))](
        x, w, normalized, out, T, H=H, K=K, STREAMS=S, BLOCK_T=bt, BLOCK_H=bh, BLOCK_K=bk,
        num_warps=warps, num_stages=stages)


def bench(fn, iters):
    for _ in range(5): fn(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters): fn(i % NCOPY)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


torch.manual_seed(0)
ws = [torch.randn(S * H, K, dtype=torch.bfloat16, device=dev) * 0.05 for _ in range(NCOPY)]
configs = [(bt, bh, 64, wp, st) for bt in (16, 32, 64) for bh in (64, 128) for wp in (4, 8) for st in (2, 3)]
print(f"{'T':>5s} {'unfused us':>11s} {'fused us':>9s} {'speedup':>8s} {'max|diff|':>10s}  best config (BT,BH,BK,warps,stages)")
for T in (5, 20, 40, 4096):
    iters = 200 if T < 1000 else 20
    xs = [torch.randn(T, K, dtype=torch.bfloat16, device=dev) for _ in range(NCOPY)]
    norms = [torch.randn(T, S * H, dtype=torch.bfloat16, device=dev) for _ in range(NCOPY)]
    logits = torch.empty(T, S * H, dtype=torch.bfloat16, device=dev)
    out_u = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    out_f = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
    t_u = bench(lambda i: unfused(xs[i], ws[i], norms[i], logits, out_u), iters)
    best = None
    for cfg in configs:
        if T < 64 and cfg[0] > 16:
            continue
        try:
            t = bench(lambda i: fused(xs[i], ws[i], norms[i], out_f, *cfg), iters)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, cfg)
    unfused(xs[0], ws[0], norms[0], logits, out_u)
    fused(xs[0], ws[0], norms[0], out_f, *best[1])
    torch.cuda.synchronize()
    diff = (out_u.float() - out_f.float()).abs().max().item()
    print(f"{T:5d} {t_u:11.1f} {best[0]:9.1f} {t_u / best[0]:7.2f}x {diff:10.4f}  {best[1]}")

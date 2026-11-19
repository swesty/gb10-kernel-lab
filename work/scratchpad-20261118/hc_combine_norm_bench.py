"""HyperConnection combine-norm: b12x PackedCombineNorm (live) vs Triton variants, GB10.

Semantics (per token t, stream s; streams=4, hidden=2560):
  scale     = 2 * sigmoid(injection_logits[t, s] / streams)
  combined  = bf16(state[t, s, :] + scale * block_output[t, :])
  normalized = bf16(combined * rsqrt(mean(combined^2) + eps) * (1 + weight[s, :]))
Variant A: one program per (token, stream), all row loads issued up front.
Variant B: one program per token for all 4 streams (block_output row loaded once).
Usage: python hc_combine_norm_bench.py
"""
import torch
import triton
import triton.language as tl

S, H, LOWRANK, DOWN_W = 4, 2560, 320, 336
EPS = 1e-6
dev = "cuda"
NCOPY = 4


@triton.jit
def _cn_a(state_ptr, out_ptr, logit_ptr, logit_stride, w_ptr, comb_ptr, norm_ptr, eps,
          H: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    row = (token * S + s) * H
    st = tl.load(state_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    ob = tl.load(out_ptr + token * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + s * H + cols, mask=mask, other=0.0).to(tl.float32)
    logit = tl.load(logit_ptr + token * logit_stride + s).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit / S)
    comb = (st + scale * ob).to(tl.bfloat16)
    tl.store(comb_ptr + row + cols, comb, mask=mask)
    r = comb.to(tl.float32)
    inv = tl.rsqrt(tl.sum(r * r, axis=0) / H + eps)
    tl.store(norm_ptr + row + cols, (r * inv * (1.0 + w)).to(tl.bfloat16), mask=mask)


@triton.jit
def _cn_b(state_ptr, out_ptr, logit_ptr, logit_stride, w_ptr, comb_ptr, norm_ptr, eps,
          H: tl.constexpr, S: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    srows = tl.arange(0, S)
    mask = cols < H
    m2 = mask[None, :] & (srows[:, None] < S)
    offs = (token * S + srows[:, None]) * H + cols[None, :]
    st = tl.load(state_ptr + offs, mask=m2, other=0.0).to(tl.float32)
    ob = tl.load(out_ptr + token * H + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + srows[:, None] * H + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    logit = tl.load(logit_ptr + token * logit_stride + srows).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit / S)
    comb = (st + scale[:, None] * ob[None, :]).to(tl.bfloat16)
    tl.store(comb_ptr + offs, comb, mask=m2)
    r = comb.to(tl.float32)
    inv = tl.rsqrt(tl.sum(r * r, axis=1) / H + eps)
    tl.store(norm_ptr + offs, (r * inv[:, None] * (1.0 + w)).to(tl.bfloat16), mask=m2)


def bench(fn, iters):
    for _ in range(5): fn(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters): fn(i % NCOPY)
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def b12x_runner(T, inputs, weight):
    from b12x.norm.hyperconnection import api, _impl
    from b12x.preparation import PreparedCall
    from b12x.preparation.session import prepare_default
    caps = api.Caps(device=dev, max_tokens=T, hidden_size=H, streams=S, lowrank=LOWRANK, dtype=torch.bfloat16)
    plan = api.plan(caps, invocation={"operation": "combine_norm", "eps": EPS})
    st0, ob0, inj0 = inputs[0]
    req = plan.request(name=f"bench-combine-norm-{T}", prepare_call=lambda state: PreparedCall(
        run=lambda: _impl.run_combine_norm_impl(st0, ob0, inj0, weight, eps=EPS, plan=state), owners=()))
    prepare_default(req)
    return lambda i: api.run_combine_norm(*inputs[i], weight, eps=EPS, plan=plan)


torch.manual_seed(0)
weight = (torch.randn(S * H, device=dev) * 0.1).to(torch.bfloat16)
print(f"{'T':>5s} {'b12x us':>8s} {'A us (cfg)':>16s} {'B us (cfg)':>16s} {'combined':>9s} {'normalized mismatches':>22s}")
for T in (5, 20, 80, 4096):
    iters = 300 if T < 1000 else 30
    inputs = []
    for _ in range(NCOPY):
        st = torch.randn(T, S * H, device=dev).to(torch.bfloat16)
        ob = torch.randn(T, H, device=dev).to(torch.bfloat16)
        down = torch.randn(T, DOWN_W, device=dev).to(torch.bfloat16)
        inputs.append((st, ob, down[:, LOWRANK:LOWRANK + S]))  # strided view, as in the live model
    run_b12x = b12x_runner(T, inputs, weight)
    t_b = bench(run_b12x, iters)
    comb = torch.empty(T, S * H, dtype=torch.bfloat16, device=dev)
    norm = torch.empty_like(comb)
    best = {}
    for name, kern, grid_fn, warps_list in (("A", _cn_a, lambda T: (T, S), (4, 8)),
                                            ("B", _cn_b, lambda T: (T,), (8, 16))):
        for wp in warps_list:
            def f(i, kern=kern, wp=wp, grid_fn=grid_fn):
                st, ob, inj = inputs[i]
                kern[grid_fn(T)](st, ob, inj, inj.stride(0), weight, comb, norm, EPS, H=H, S=S, BLOCK=4096,
                                 num_warps=wp)
            t = bench(f, iters)
            if name not in best or t < best[name][0]:
                best[name] = (t, wp, f)
    ref_c, ref_n = run_b12x(0)
    best["A"][2](0); torch.cuda.synchronize()
    cmis = (comb != ref_c).sum().item()
    nmis = (norm != ref_n).sum().item()
    nmax = (norm.float() - ref_n.float()).abs().max().item()
    print(f"{T:5d} {t_b:8.1f} {best['A'][0]:9.1f} (w{best['A'][1]:2d}) {best['B'][0]:9.1f} (w{best['B'][1]:2d}) "
          f"{cmis:9d} {nmis:10d} of {norm.numel()} (max {nmax:.4f})")

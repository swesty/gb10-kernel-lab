"""Cost prototype for 2-row QSA attention using the existing b12x kernel unchanged.

A pair of adjacent query rows becomes one kernel "row" with 24 query heads per KV head (12 per real row), so
the kernel picks its 32-row-tile / 8-warp configuration. The pair attends to the UNION of both rows' real
selections (from live Flash Next dumps). No per-row masking, so outputs are not exact QSA; this measures cost.

Compares, on the same live selections and the same synthetic KV cache:
  per-row:  rows x (24 q heads, 2 kv heads), selection width 2051
  pair:     rows/2 x (48 q heads, 2 kv heads), selection width = max union size (rounded up to 32)
Usage: python pair_proto.py /dump/sel_031.pt [--iters 20]
"""
import argparse
import math

import torch

from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
from b12x.attention.paged._selected_forward_config import HEAD_DIM, SELECTION_WIDTH

QH, KVH, D, PAGE = 24, 2, HEAD_DIM, 3024


def timed(fn, iters):
    fn(); torch.cuda.synchronize()
    for _ in range(3): fn()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--width", choices=("max", "mean"), default="max",
                    help="kernel selection width: max union (exact work + padding) or mean union (truncated, ~fair work)")
    a = ap.parse_args()
    dev = "cuda"
    d = torch.load(a.dump)
    sel = d["selected"].to(dev).to(torch.int32)
    qpos = d["qpos"].to(dev).to(torch.int64)
    rows = sel.shape[0] - sel.shape[0] % 2
    sel, qpos = sel[:rows], qpos[:rows]
    context = int(qpos.max()) + 1
    g = torch.Generator(device=dev).manual_seed(0)
    pages = math.ceil(context / PAGE)
    cache = (torch.randn(2, pages + 3, PAGE, KVH, D, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    block_table = torch.randperm(pages + 3, device=dev, generator=g)[:pages].to(torch.int32).view(1, -1).contiguous()
    one = torch.tensor([0.75], dtype=torch.float32, device=dev)
    scale = 1.0 / math.sqrt(D)
    common = dict(key_cache=cache[0], value_cache=cache[1], k_descale=one, v_descale=one, block_table=block_table)

    # per-row baseline
    q1 = torch.randn(rows, QH, D, device=dev, generator=g).to(torch.bfloat16)
    out1 = torch.empty_like(q1)
    lse1 = torch.empty(rows, QH, dtype=torch.float32, device=dev)
    rid1 = torch.zeros(rows, dtype=torch.int32, device=dev)
    t_row = timed(lambda: launch_selected_paged_gqa_direct(
        query=q1, request_ids=rid1, selected_positions=sel, query_positions=qpos, output=out1, output_lse=lse1,
        softmax_scale=scale, kv_warps=2, **common), a.iters)

    # pair unions (sorted unique positions per pair)
    pairs = rows // 2
    unions = []
    for p in range(pairs):
        u = torch.cat((sel[2 * p], sel[2 * p + 1]))
        unions.append(torch.unique(u[u >= 0]))
    sizes = torch.tensor([u.numel() for u in unions], dtype=torch.float32)
    target = sizes.max() if a.width == "max" else sizes.mean()
    width = max(int(math.ceil(float(target) / 32) * 32), SELECTION_WIDTH)
    psel = torch.full((pairs, width), -1, dtype=torch.int32, device=dev)
    for p, u in enumerate(unions):
        u = u[:width]
        psel[p, :u.numel()] = u.to(torch.int32)
    pqpos = qpos[1::2].contiguous()  # the later row of each pair bounds causality
    # 48 heads per pair: kv head k owns heads [24k, 24k+24) = row A's 12 heads of k, then row B's 12 heads of k
    qa, qb = q1[0::2].view(pairs, KVH, 12, D), q1[1::2].view(pairs, KVH, 12, D)
    q2 = torch.cat((qa, qb), dim=2).reshape(pairs, KVH * 24, D).contiguous()
    out2 = torch.empty_like(q2)
    lse2 = torch.empty(pairs, KVH * 24, dtype=torch.float32, device=dev)
    rid2 = torch.zeros(pairs, dtype=torch.int32, device=dev)
    t_pair = timed(lambda: launch_selected_paged_gqa_direct(
        query=q2, request_ids=rid2, selected_positions=psel, query_positions=pqpos, output=out2, output_lse=lse2,
        softmax_scale=scale, kv_warps=2, **common), a.iters)

    per_row = (sel >= 0).sum(1).float().mean().item()
    print(f"{a.dump}: rows {rows} context {context} mean selection {per_row:.0f}, pair union mean {sizes.mean():.0f} "
          f"(x{sizes.mean() / per_row:.2f}), max {int(sizes.max())}, kernel width {width}")
    print(f"  per-row kernel : {t_row:9.1f} us")
    print(f"  pair kernel    : {t_pair:9.1f} us   ({t_row / t_pair:.2f}x vs per-row)")


if __name__ == "__main__":
    main()

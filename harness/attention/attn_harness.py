"""Standalone harness for b12x selected-position paged attention (Qwen3.8 Flash Next shapes).

Calls b12x.attention.paged._selected_forward.launch_selected_paged_gqa_direct, the kernel live prefill uses
(rows > MAX_SPLIT_ROWS), checks sampled rows against an FP32 reference, and times it per kv_warps.

Live geometry (from the :8112 engine config and the Nsight Compute capture, 2026-11-18):
  24 query heads, 2 KV heads, head dim 256, FP8 E4M3 KV cache, 3024-token pages, up to 2051 selected
  positions per query row, 4096-row prefill chunks, kv_warps=2 (42.56 KB dynamic smem per 128-thread block).

Usage: python attn_harness.py [--rows 4096] [--context 16384] [--kv-warps 1,2,4] [--iters 50]
"""
import argparse
import math

import torch

from b12x.attention.paged._selected_forward import launch_selected_paged_gqa_direct
from b12x.attention.paged._selected_forward_config import HEAD_DIM, SELECTION_WIDTH

QH, KVH, D = 24, 2, HEAD_DIM
GROUP = QH // KVH


COMPRESS_RATIO = 4                      # QSA scores groups of 4 contiguous tokens
BUDGET = SELECTION_WIDTH - COMPRESS_RATIO + 1  # 2048 tokens = 512 groups; width adds a <=3-token tail


def qsa_like_selection(query_positions, g, dev, row_noise):
    """Mimic b12x score_select_reference: top-512 groups of 4 in descending score order, then the causal tail.
    Rows share a base group score plus per-row noise, so neighbouring rows pick overlapping groups."""
    rows = query_positions.numel()
    max_groups = (int(query_positions.max()) + 1) // COMPRESS_RATIO
    base = torch.randn(max_groups, device=dev, generator=g)
    selected = torch.full((rows, SELECTION_WIDTH), -1, dtype=torch.int32, device=dev)
    ar = torch.arange(COMPRESS_RATIO, device=dev)
    for r in range(rows):
        p = int(query_positions[r])
        eligible = (p + 1) // COMPRESS_RATIO
        count = min(BUDGET // COMPRESS_RATIO, eligible)
        parts = []
        if count:
            scores = base[:eligible] + row_noise * torch.randn(eligible, device=dev, generator=g)
            groups = torch.topk(scores, count, sorted=True).indices
            parts.append((groups[:, None] * COMPRESS_RATIO + ar[None, :]).flatten())
        parts.append(torch.arange(eligible * COMPRESS_RATIO, p + 1, device=dev))
        pos = torch.cat(parts)
        selected[r, :pos.numel()] = pos.to(torch.int32)
    return selected


def build(rows, context, page, seed, descale, dev, pattern="qsa", row_noise=0.3):
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = math.ceil(context / page)
    num_pages = pages + 3
    # Same-strided K and V views of one allocation, like a [2, pages, page, heads, dim] vLLM cache.
    cache = (torch.randn(2, num_pages, page, KVH, D, device=dev, generator=g) * 0.5).to(torch.float8_e4m3fn)
    key_cache, value_cache = cache[0], cache[1]
    block_table = torch.randperm(num_pages, device=dev, generator=g)[:pages].to(torch.int32).view(1, -1).contiguous()
    query_positions = torch.arange(context - rows, context, dtype=torch.int64, device=dev)  # last chunk of the prompt
    request_ids = torch.zeros(rows, dtype=torch.int32, device=dev)
    if pattern == "qsa":
        selected = qsa_like_selection(query_positions, g, dev, row_noise)
    else:  # "random": independent random positions per row, sorted (worst case for locality)
        selected = torch.full((rows, SELECTION_WIDTH), -1, dtype=torch.int32, device=dev)
        for r in range(rows):
            p = int(query_positions[r])
            n = min(p + 1, SELECTION_WIDTH)
            pos = torch.randperm(p + 1, device=dev, generator=g)[:n].sort().values
            selected[r, :n] = pos.to(torch.int32)
    query = torch.randn(rows, QH, D, device=dev, generator=g).to(torch.bfloat16)
    k_descale = torch.tensor([descale], dtype=torch.float32, device=dev)
    v_descale = torch.tensor([descale], dtype=torch.float32, device=dev)
    return dict(query=query, key_cache=key_cache, value_cache=value_cache, k_descale=k_descale, v_descale=v_descale,
                block_table=block_table, request_ids=request_ids, selected_positions=selected,
                query_positions=query_positions)


def reference(inp, rows_to_check, page, scale):
    """FP32 attention over each row's selected positions, GQA head h -> KV head h // GROUP."""
    out, lse = {}, {}
    bt = inp["block_table"][0].long()
    for r in rows_to_check:
        pos = inp["selected_positions"][r]
        pos = pos[pos >= 0].long()
        phys, off = bt[pos // page], pos % page
        q = inp["query"][r].float()  # [QH, D]
        o_r, l_r = [], []
        for kvh in range(KVH):
            k = inp["key_cache"][phys, off, kvh].float() * inp["k_descale"][0]
            v = inp["value_cache"][phys, off, kvh].float() * inp["v_descale"][0]
            s = q[kvh * GROUP:(kvh + 1) * GROUP] @ k.T * scale  # [GROUP, n]
            l_r.append(torch.logsumexp(s, dim=-1))
            o_r.append(torch.softmax(s, dim=-1) @ v)
        out[r], lse[r] = torch.cat(o_r), torch.cat(l_r)
    return out, lse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--page", type=int, default=3024)
    ap.add_argument("--kv-warps", default="1,2,4")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--check-rows", type=int, default=24)
    ap.add_argument("--descale", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile-one", type=int, default=0, help="kv_warps to run once inside cudaProfilerStart/Stop")
    ap.add_argument("--split", action="store_true", help="test the decode split path (rows <= 64) + merge")
    ap.add_argument("--pattern", choices=("qsa", "random"), default="qsa")
    ap.add_argument("--row-noise", type=float, default=0.3, help="qsa pattern: per-row score noise vs shared base")
    a = ap.parse_args()
    dev = "cuda"
    scale = 1.0 / math.sqrt(D)
    inp = build(a.rows, a.context, a.page, a.seed, a.descale, dev, a.pattern, a.row_noise)
    print(f"pattern={a.pattern} row_noise={a.row_noise}")
    n_sel = (inp["selected_positions"] >= 0).sum().item()
    rows_to_check = sorted(set(torch.linspace(0, a.rows - 1, a.check_rows).long().tolist()))
    ref_out, ref_lse = reference(inp, rows_to_check, a.page, scale)
    gathered = n_sel * KVH * D * 2  # bytes of FP8 K+V gathered (each (row, kv head) block reads its own rows)
    flops = 4.0 * n_sel * QH * D    # QK^T and PV, 2 flops per MAC
    print(f"rows={a.rows} context={a.context} page={a.page} selected/row avg={n_sel / a.rows:.0f} "
          f"gathered KV={gathered / 1e9:.2f} GB per call  flops={flops / 1e12:.3f} TFLOP per call")
    print(f"{'kv_warps':>8s} {'us/call':>9s} {'gathered GB/s':>13s} {'TFLOP/s':>8s} {'max|err|':>9s} {'max rel':>8s} {'lse err':>8s}")
    kw_list = [int(x) for x in a.kv_warps.split(",")]
    if a.split:
        kw_list = [2]  # the split path always compiles kv_warps=2
    for kw in kw_list:
        out = torch.empty(a.rows, QH, D, dtype=torch.bfloat16, device=dev)
        lse = torch.empty(a.rows, QH, dtype=torch.float32, device=dev)
        if a.split:
            from b12x.attention.paged._selected_forward import launch_sparse_gqa_split, launch_sparse_gqa_merge
            splits = 64 if a.rows == 1 else 32 if a.rows <= 4 else 16  # b12x QSA contract for rows <= 64
            p_out = torch.empty(a.rows, splits, QH, D, dtype=torch.float32, device=dev)
            p_lse = torch.empty(a.rows, splits, QH, dtype=torch.float32, device=dev)
            def run():
                launch_sparse_gqa_split(**inp, partial_output=p_out, partial_lse=p_lse, softmax_scale=scale, splits=splits)
                launch_sparse_gqa_merge(partial_output=p_out, partial_lse=p_lse, output=out, output_lse=lse,
                                        rows=a.rows, splits=splits)
        else:
            def run():
                launch_selected_paged_gqa_direct(**inp, output=out, output_lse=lse, softmax_scale=scale, kv_warps=kw)
        run(); torch.cuda.synchronize()  # compile + warm
        err = rel = lerr = 0.0
        for r in rows_to_check:
            d = (out[r].float() - ref_out[r]).abs()
            err = max(err, d.max().item())
            rel = max(rel, (d / ref_out[r].abs().clamp_min(1e-2)).max().item())
            lerr = max(lerr, min((lse[r] - ref_lse[r]).abs().max().item(),                    # natural log
                                 (lse[r] - ref_lse[r] / math.log(2)).abs().max().item()))   # or log2
        for _ in range(5): run()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(a.iters): run()
        t1.record(); torch.cuda.synchronize()
        us = t0.elapsed_time(t1) * 1000 / a.iters
        print(f"{kw:8d} {us:9.1f} {gathered / us / 1e3:13.1f} {flops / us / 1e6:8.2f} {err:9.4f} {rel:8.3f} {lerr:8.4f}")
        if a.profile_one == kw:
            torch.cuda.cudart().cudaProfilerStart(); run(); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()

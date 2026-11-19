"""Summarize dense_gemm_harness.py sweep JSON files: per shape, the stock default vs the best stock tactic vs
the best tactic under each scheduler-raster experiment. Usage: python3 dense_gemm_summarize.py DIR"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
rows = []
for f in sorted(glob.glob(os.path.join(d, "sw8_*.json"))):
    for r in json.load(open(f)):
        if "med_us" in r:
            r["_file"] = os.path.basename(f)
            rows.append(r)


def label(r):
    if r.get("default"):
        return "default(64x128,bk128)"
    return f"{r['tile'][0]}x{r['tile'][1]} bk{r['tile_k']} swap={int(r['swap_ab'])} lmu={r['large_m_unroll']}"


for shape in ("qkvz", "ba", "out", "gu", "down", "attn_qkv"):
    rs = [r for r in rows if r["shape"] == shape]
    if not rs:
        continue
    print(f"== {shape}  M={rs[0]['M']} N={rs[0]['N']} K={rs[0]['K']}")
    for sched in sorted({r.get("sched", "stock") for r in rs}):
        sr = [r for r in rs if r.get("sched", "stock") == sched]
        dflt = [r for r in sr if r.get("default")]
        best = sorted(sr, key=lambda r: r["med_us"])[:3]
        if dflt:
            r = dflt[0]
            print(f"  [{sched:9s}] default            med {r['med_us']:7.1f} us  (placements {r['min_us']:.0f}-{r['max_us']:.0f})  kern(p0) {r['kern_cold_p0_us']:.1f}")
        for r in best:
            print(f"  [{sched:9s}] {label(r):34s} med {r['med_us']:7.1f} us  (placements {r['min_us']:.0f}-{r['max_us']:.0f})  kern(p0) {r['kern_cold_p0_us']:.1f}  err {r['err_vs_qref']:.2e} vs-live {r.get('err_vs_live', float('nan')):.1e}")

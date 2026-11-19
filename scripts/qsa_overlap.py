"""Overlap of QSA selections between neighbouring query rows (live Flash Next dumps).

For a block of k consecutive rows, a multi-row kernel would gather the union of their selected positions
once and mask per row. Useful fraction = (sum of per-row selection sizes) / (k * |union|): the share of
QK/PV work in such a kernel that a per-row kernel would also do. Expansion = |union| / mean per-row size.
"""
import glob
import sys

import torch

paths = sorted(glob.glob(sys.argv[1] + "/sel_*.pt"))
print(f"{len(paths)} dumps")
for path in paths:
    d = torch.load(path)
    sel, qpos = d["selected"].long(), d["qpos"].long()
    rows = sel.shape[0]
    valid = sel >= 0
    sizes = valid.sum(1).float()
    line = [f"call {d['call']:3d} rows {rows} qpos {int(qpos.min())}-{int(qpos.max())} mean sel {sizes.mean():.0f}"]
    # adjacent-row Jaccard on positions
    jac = []
    for r in range(0, rows - 1, max(1, rows // 256)):
        a = set(sel[r][valid[r]].tolist()); b = set(sel[r + 1][valid[r + 1]].tolist())
        jac.append(len(a & b) / max(1, len(a | b)))
    line.append(f"adjacent Jaccard {sum(jac) / len(jac):.3f}")
    for k in (2, 4, 8, 16):
        useful, expand = [], []
        for start in range(0, rows - k + 1, max(k, rows // 128)):
            block = sel[start:start + k]
            u = torch.unique(block[block >= 0]).numel()
            s = sizes[start:start + k]
            useful.append(float(s.sum()) / (k * u))
            expand.append(u / float(s.mean()))
        line.append(f"k={k}: union x{sum(expand) / len(expand):.2f} useful {100 * sum(useful) / len(useful):.0f}%")
    print("  ".join(line))

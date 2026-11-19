import csv, collections, sys
O = "/Users/swesty/Developer/kernel-engineering/Kernel-Engg/output/ncu-flashnext-20261118"
FAM = [("SelectedPositionPagedForward","paged attention"),("RepresentativeScore","qsa score"),("PackedRecurrentQwen","GDN decode"),
       ("MoEDynamicKernelSilu","MoE SiLU"),("PackedCombineNorm","combine-norm"),("_gate_mean_kernel","_gate_mean"),
       ("libdense_gemmDenseGemm","dense GEMM"),("wmma_tensorop_bf16","WMMA bf16"),("gemvx","gemv bf16")]
def fam(n):
    for k, l in FAM:
        if k in n: return l
    return n[:24]
WANT = {"Duration": "dur", "Memory Throughput": "memsol", "Compute (SM) Throughput": "smsol", "Achieved Occupancy": "occ",
        "Theoretical Occupancy": "theo", "Registers Per Thread": "regs", "Dynamic Shared Memory Per Block": "dsmem",
        "Warp Cycles Per Issued Instruction": "cpi", "Waves Per SM": "waves", "Grid Size": "grid", "Block Size": "block"}
def load(rep):
    det = collections.defaultdict(dict); names = {}; units = {}
    for r in csv.DictReader(open(f"{O}/{rep}-details.csv")):
        m = r["Metric Name"]
        if m in WANT and WANT[m] not in det[r["ID"]]:
            try: v = float(r["Metric Value"].replace(",", ""))
            except ValueError: continue
            u = r["Metric Unit"]
            if m == "Duration": v = v / 1000 if u == "ns" else v * 1000 if u == "ms" else v  # -> us
            if m == "Dynamic Shared Memory Per Block": v = v * 1024 if u == "Kbyte/block" else v / 1 if u == "byte/block" else v
            det[r["ID"]][WANT[m]] = v
        names[r["ID"]] = r["Kernel Name"]
    raw = list(csv.DictReader(open(f"{O}/{rep}-raw.csv")))[1:]
    stalls = [c for c in raw[0] if c.startswith("smsp__average_warps_issue_stalled_") and c.endswith("_per_issue_active.ratio")]
    for r in raw:
        d = det[r["ID"]]
        f = float(r["lts__d_sectors_fill_device.sum"].replace(",", "") or 0) + float(r["lts__d_sectors_fill_sysmem.sum"].replace(",", "") or 0)
        d["bytes"] = f * 32
        d["tensor"] = float(r["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"].replace(",", "") or 0)
        st = {c.replace("smsp__average_warps_issue_stalled_", "").replace("_per_issue_active.ratio", ""): float(r[c].replace(",", "") or 0) for c in stalls}
        d["stalls"] = st
    return det, names
for rep in ("main", "wmma"):
    det, names = load(rep)
    ids = sorted(det, key=int)
    pre = min((int(i) for i in ids if fam(names[i]) == "paged attention" and det[i].get("dur", 0) > 1000), default=10**9)
    groups = collections.defaultdict(list)
    for i in ids:
        d = det[i]
        if "dur" not in d: continue
        groups[(fam(names[i]), "prefill" if int(i) >= pre else "decode")].append(d)
    print(f"\n### {rep} (prefill starts at ID {pre})")
    print(f"{'kernel':16s} {'phase':7s} {'n':>3s} {'dur us':>8s} {'GB/s':>6s} {'mem%':>5s} {'sm%':>5s} {'tens%':>6s} {'occ%':>5s}/{'theo':<4s} {'regs':>4s} {'smemKB':>6s} {'waves':>5s}  top stalls (cycles/issue)")
    for (f, ph), L in sorted(groups.items()):
        a = lambda k: sum(x.get(k, 0) for x in L) / len(L)
        gbs = sum(x["bytes"] for x in L) / sum(x["dur"] for x in L) / 1e3
        st = collections.Counter()
        for x in L: st.update(x.get("stalls", {}))
        top = ", ".join(f"{k} {v/len(L):.1f}" for k, v in st.most_common(3))
        print(f"{f:16s} {ph:7s} {len(L):3d} {a('dur'):8.1f} {gbs:6.0f} {a('memsol'):5.0f} {a('smsol'):5.0f} {a('tensor'):6.1f} {a('occ'):5.0f}/{a('theo'):<4.0f} {a('regs'):4.0f} {a('dsmem')/1024:6.1f} {a('waves'):5.2f}  {top}")

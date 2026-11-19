import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select k.start, k.end, k.gridX, k.gridY, k.gridZ, k.blockX, k.dynamicSharedMemory, k.registersPerThread, s.value
  from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start""").fetchall()
moe = [r for r in rows if "MoEDynamicKernelSilu" in r[8]]
t0, t1 = moe[0][0], moe[-1][1]
agg = collections.defaultdict(lambda: [0, 0])
for st, en, gx, gy, gz, bx, sm, reg, name in rows:
    if st < t0 or en > t1: continue
    if "DenseGemm" in name or "dense_gemm" in name.lower() or "Gemm" in name:
        short = name.replace("kernel_cutlass_kernel_", "")[:70]
        key = (short, gx, gy, gz, bx, sm, reg)
        agg[key][0] += 1; agg[key][1] += en - st
tot = sum(v[1] for v in agg.values())
print(f"prefill window {(t1-t0)/1e6:.0f} ms, GEMM-like kernels {tot/1e6:.0f} ms")
for k, (n, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
    print(f"{t/1e6:8.1f} ms  n={n:4d}  avg {t/n/1e3:7.1f} us  grid=({k[1]},{k[2]},{k[3]}) blk={k[4]} smem={k[5]} reg={k[6]}  {k[0]}")

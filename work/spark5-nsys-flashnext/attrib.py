import sqlite3, sys, bisect, re, collections
db = sqlite3.connect(sys.argv[1])
ids = {n: m for m, n in db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")}
def series(name):
    rows = db.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp", (ids[name],)).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]
mt, miss = series("L2 Read Misses (memory reads) [GB/s]")
_, smact = series("SMs Active [Throughput %]")
_, tens = series("Tensor Active [Throughput %]")
ks = db.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
def short(n):
    n = n.replace("kernel_cutlass_kernel_", "").replace("void ", "")
    for key, lab in [("MoEDynamicKernelSilu","MoE SiLU dynamic (b12x)"),("W4A16FusedMoe","MoE W4A16 fused (b12x, draft?)"),("libdense_gemmDenseGemm","dense GEMM (b12x libdense)"),
                     ("wmma_tensorop_bf16","cuBLAS WMMA bf16 GEMM (sm80)"),("gemvx","cuBLAS gemv bf16"),("nvjet_sm121","cuBLAS nvjet sm121 GEMM"),
                     ("PackedRecurrentQwen","GDN decode (b12x)"),("SelectedPositionPagedForward","paged attention (b12x)"),("RepresentativeScore","attention qsa score (b12x)"),
                     ("PackedCombineNorm","combine-norm (b12x)"),("MTPPrefillBf16Gemm","MTP prefill bf16 GEMM"),("RecurrenceKernel","GDN prefill recurrence"),("PrepareKernel","GDN prefill prepare"),
                     ("_gate_mean_kernel","_gate_mean (triton)"),("vectorized_gather","torch gather"),("s16816gemm","cutlass sm80 tensorop GEMM")]:
        if key in n: return lab
    return re.sub(r"[<(].*", "", n)[:40]
agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])  # samples, miss sum, sm sum, tensor sum, kernel time
tot_time = 0
for s, e, n in ks:
    a = bisect.bisect_left(mt, s); b = bisect.bisect_right(mt, e)
    g = agg[short(n)]; g[4] += e - s; tot_time += e - s
    for i in range(a, b):
        g[0] += 1; g[1] += miss[i]; g[2] += smact[i]; g[3] += tens[i]
span = (ks[-1][1] - ks[0][0])
a = bisect.bisect_left(mt, ks[0][0]); b = bisect.bisect_right(mt, ks[-1][1])
print(f"{sys.argv[1]}: window {span/1e9:.2f} s, whole-window memory read {sum(miss[a:b])/(b-a)*32e-9:.0f} GB/s")
print("  %-34s %6s %9s %7s %7s %8s  samples" % ("kernel", "time%", "mem GB/s", "%of273", "SMact%", "tensor%"))
for k, (n, m, smv, tv, t) in sorted(agg.items(), key=lambda x: -x[1][4])[:12]:
    if n == 0: print(f"  {k:34s} {100*t/tot_time:6.1f}   (no samples: kernels shorter than 50 us)"); continue
    gb = m / n * 32e-9
    print(f"  {k:34s} {100*t/tot_time:6.1f} {gb:9.0f} {100*gb/273:7.0f} {smv/n:7.0f} {tv/n:8.1f}  {n}")

import sqlite3, sys, re, collections
def cat(n):
    for key, lab in [("MoEDynamicKernelSilu","MoE (b12x)"),("libdense_gemmDenseGemm","dense GEMM MXFP8 (b12x)"),("MTPPrefillBf16Gemm","MTP feedback GEMM (b12x)"),("nvjet_sm121","cuBLAS GEMM (HC/MTP)"),
        ("Kernel2","cuBLAS GEMM (HC/MTP)"),("wmma_tensorop","cuBLAS GEMM (HC/MTP)"),
        ("SelectedPositionPagedForward","attention selected (b12x)"),("RepresentativeScore","attention QSA score"),("Qsa","attention QSA score"),
        ("PackedCombineNorm","HC combine-norm"),("_hc_combine_norm","HC combine-norm"),("_gate_mean","HC gate+mean"),("_hc_fused_up","HC gate+mean"),("ScaledSilu","HC other"),("_grouped_rmsnorm","HC other"),
        ("RecurrenceKernel","GDN core"),("PrepareKernel","GDN core"),("PrologueKernel","GDN core"),("causal_conv1d","GDN conv"),("layer_norm_fwd","GDN gated norm"),
        ("_scatter_b12x_gdn","GDN gather/scatter/fill"),("vectorized_gather","GDN gather/scatter/fill"),
        ("_quantize","activation quantize"),("topkGating","router"),
        ("elementwise","torch elementwise/fill"),("reduce_kernel","torch reduce"),("Memcpy","copy")]:
        if key in n: return lab
    return "other: " + re.sub(r"[<(].*", "", n)[:40]
for f in sys.argv[1:]:
    db = sqlite3.connect(f)
    ks = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
    # prefill = longest burst: take kernels between first and last MoE launch with duration > 2 ms
    big = [i for i,(s,e,n) in enumerate(ks) if "MoEDynamicKernelSilu" in n and e-s > 2e6]
    sel = ks[big[0]-200: big[-1]+60]
    agg = collections.Counter(); tot = 0
    for s,e,n in sel: agg[cat(n)] += e-s; tot += e-s
    span = (sel[-1][1]-sel[0][0])/1e6
    print(f"== {f}: prefill window {span:.0f} ms, kernel sum {tot/1e6:.0f} ms")
    for k,v in agg.most_common(22): print(f"  {k:36s} {v/1e6:8.1f} ms {100*v/tot:5.1f}%")

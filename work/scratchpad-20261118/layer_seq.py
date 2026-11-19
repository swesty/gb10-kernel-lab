import sqlite3, sys, re
db = sqlite3.connect(sys.argv[1])
rows = db.execute("""select k.start, k.end, k.gridX, k.gridY, k.gridZ, k.blockX, k.dynamicSharedMemory, s.value
  from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start""").fetchall()
moe = [i for i, r in enumerate(rows) if "MoEDynamicKernelSilu" in r[7]]
def short(n):
    n = n.replace("kernel_cutlass_kernel_", "").replace("void ", "")
    for key in ["MoEDynamicKernelSilu","DenseGemmKernel","MTPPrefillBf16Gemm","PackedCombineNorm","RecurrenceKernel","PrepareKernel",
                "SelectedPositionPagedForward","RepresentativeScore","_hc_fused_up_gated_mean","_combine_norm","causal_conv1d","layer_norm","nvjet","wmma","gemv",
                "_qsa_qk_norm_rope","_qsa_sigmoid_gate","topk","quantize","elementwise","reduce_kernel"]:
        if key in n: return key
    return re.sub(r"[<(].*", "", n)[:40]
# print two full layers (between MoE launches 4..6) of chunk 2
for a, b in [(moe[60], moe[62])]:
    for r in rows[a:b+1]:
        print(f"{(r[1]-r[0])/1e3:8.1f} us grid=({r[2]},{r[3]},{r[4]}) blk={r[5]} smem={r[6]}  {short(r[7])}")

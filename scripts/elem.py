import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
ks = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
big = [i for i,(s,e,n) in enumerate(ks) if "MoEDynamicKernelSilu" in n and e-s > 2e6]
sel = ks[big[0]-200: big[-1]+60]
def short(n):
    n = n.replace("kernel_cutlass_kernel_", "")
    return re.sub(r"_object_at_.*|[<(].*", "", n)[:38]
def functor(n):
    m = re.findall(r"(FillFunctor|CUDAFunctor_add|CUDAFunctorOnSelf_add|MulFunctor|AddFunctor|sigmoid|copy_kernel|direct_copy|BinaryFunctor<[^,>]*|\w+_kernel_cuda|neg|exp|rsqrt|pow|div|where|clamp|silu|bfloat16_copy)", n)
    return "/".join(dict.fromkeys(m))[:50] or n[:60]
agg = collections.Counter(); cnt = collections.Counter()
for i,(s,e,n) in enumerate(sel):
    if "elementwise" in n:
        prev = short(sel[i-1][2]); nxt = short(sel[i+1][2]) if i+1 < len(sel) else "END"
        key = (functor(n), prev, nxt)
        agg[key] += e-s; cnt[key] += 1
tot = sum(agg.values())
print(f"elementwise total {tot/1e6:.1f} ms")
for k,v in agg.most_common(14): print(f"{v/1e6:7.1f} ms x{cnt[k]:5d}  {k[0]:45s} | after {k[1]} | before {k[2]}")

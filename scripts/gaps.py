import sqlite3, sys, re, collections
db = sqlite3.connect(sys.argv[1])
ks = db.execute("select k.start,k.end,s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName order by k.start").fetchall()
mc = []
try:
    mc = db.execute("select start,end,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
except Exception as e: print("no memcpy table", e)
big = [i for i,(s,e,n) in enumerate(ks) if "MoEDynamicKernelSilu" in n and e-s > 2e6]
a, b = big[0], big[-1]
# extend to the first kernel of the chunk containing the first MoE: walk back while gaps < 5 ms
while a > 0 and ks[a][0] - ks[a-1][1] < 5e6: a -= 1
while b < len(ks)-1 and ks[b+1][0] - ks[b][1] < 5e6: b += 1
sel = ks[a:b+1]
t0 = sel[0][0]
def short(n): return re.sub(r"[<(].*", "", n.replace("kernel_cutlass_kernel_", "").replace("void ", ""))[:38]
gaps = []; busy_end = sel[0][1]; ksum = 0
for i in range(1, len(sel)):
    s, e, n = sel[i]
    g = s - busy_end
    if g > 0: gaps.append((g, i))
    busy_end = max(busy_end, e)
for s,e,n in sel: ksum += e - s
span = sel[-1][1] - t0
print(f"prefill burst: {len(sel)} kernels, span {span/1e6:.0f} ms, kernel sum {ksum/1e6:.0f} ms, idle {sum(g for g,_ in gaps)/1e6:.0f} ms")
hist = collections.Counter()
for g,_ in gaps:
    k = "<2us" if g<2e3 else "2-10us" if g<1e4 else "10-100us" if g<1e5 else "0.1-1ms" if g<1e6 else ">1ms"
    hist[k] += g
for k in ["<2us","2-10us","10-100us","0.1-1ms",">1ms"]:
    print(f"  gaps {k:8s}: {hist[k]/1e6:7.1f} ms  ({sum(1 for g,_ in gaps if (g<2e3 if k=='<2us' else 2e3<=g<1e4 if k=='2-10us' else 1e4<=g<1e5 if k=='10-100us' else 1e5<=g<1e6 if k=='0.1-1ms' else g>=1e6))} gaps)")
print("largest gaps (ms at, gap ms, previous kernel -> next kernel):")
for g,i in sorted(gaps, reverse=True)[:15]:
    print(f"  t={(sel[i][0]-t0)/1e6:8.1f}  gap {g/1e6:7.2f}  {short(sel[i-1][2])} -> {short(sel[i][2])}")
# which kernel precedes the 10us-1ms gaps most often (total)
pre = collections.Counter()
for g,i in gaps:
    if g >= 1e4: pre[short(sel[i-1][2]) + " -> " + short(sel[i][2])] += g
print("gap totals >=10us by kernel pair:")
for k,v in pre.most_common(12): print(f"  {v/1e6:7.1f} ms  {k}")
print("memcpy in burst:", sum(1 for s,e,_,_ in mc if t0 <= s <= sel[-1][1]), "copies,", sum(e-s for s,e,_,_ in mc if t0 <= s <= sel[-1][1])/1e6, "ms")

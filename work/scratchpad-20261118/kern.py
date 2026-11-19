import csv, sys, re
def cat(n):
    l = n.lower()
    rules = [("attention(full)", r"qsa|flash|fmha|attn_fwd|paged_attention|attention(?!.*gdn)"),
             ("gdn/linear-attn", r"gdn|gated_delta|delta_rule|chunk_|recurrent|causal_conv|conv1d|ssm|mamba|fused_sigmoid_gating|l2norm"),
             ("moe", r"moe|expert|topk|router|grouped|routing|fp4_moe|permute"),
             ("gemm/linear", r"gemm|cutlass|sm\d+_xmma|nvjet|matmul|mxfp8|linear|blockscaled|cublas"),
             ("norm/act/quant", r"rms|norm|silu|swiglu|act_and_mul|quant|fp8|fp4|gelu|sigmoid"),
             ("sampling/spec", r"sampl|argmax|softmax|top_p|top_k|reject|spec|draft|gather|index_select"),
             ("rope/kv-write", r"rotary|rope|reshape_and_cache|cache"),
             ("copy/elementwise", r"copy|elementwise|fill|cat|vectorized|reduce|memcpy|unrolled")]
    for name, pat in rules:
        if re.search(pat, l): return name
    return "other"
for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    tot = sum(float(r["Total Time (ns)"]) for r in rows)
    agg = {}
    for r in rows:
        c = cat(r["Name"]); agg.setdefault(c, [0, 0]); agg[c][0] += float(r["Total Time (ns)"]); agg[c][1] += int(r["Instances"])
    print(f"\n### {f}  total GPU kernel time {tot/1e6:.1f} ms, {sum(int(r['Instances']) for r in rows)} launches")
    for c, (t, n) in sorted(agg.items(), key=lambda x: -x[1][0]):
        print(f"  {c:18s} {100*t/tot:5.1f}%  {t/1e6:8.1f} ms  {n:7d} launches")
    print("  top kernels:")
    for r in rows[:14]:
        print(f"   {float(r['Time (%)']):5.1f}%  {int(r['Instances']):6d}x  avg {float(r['Avg (ns)'])/1e3:8.1f} us  [{cat(r['Name'])}]  {r['Name'][:110]}")

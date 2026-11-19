import json
def cfg(x):
    if x.get("default"): return "default"
    return "%dx%d:%s:%d:%d" % (x["tile"][0], x["tile"][1], x["tile_k"], bool(x["swap_ab"]), bool(x["large_m_unroll"]))
for S in ("stock", "rm=0,sw=1"):
    r = json.load(open(f"dense_sweep_{S}.json"))
    for x in r:
        print(f'{S:10s} {x["shape"]:9s} N={x["N"]:6d} K={x["K"]:5d} {cfg(x):16s} med={x.get("med_us")} min={x.get("min_us")} max={x.get("max_us")} cold={x.get("kern_cold_p0_us")} tf={x.get("tflops_med")} err={x.get("err_vs_qref", 0):.5f} {x.get("skip") or x.get("error") or ""}')
for x in json.load(open("dense_default.json")):
    print("DEFAULT", json.dumps(x)[:600])

import json, glob, collections
f = sorted(glob.glob("/root/.cache/b12x/compile/preparation/*.json"), key=lambda p: -len(open(p).read()))[0]
d = json.load(open(f)); recs = d["records"]
v0 = next(iter(recs.values())); print("value keys:", list(v0.keys()))
c = collections.Counter()
for k, v in recs.items():
    a = v.get("assignment", {})
    if a.get("backend") in ("dynamic", "micro", "w4a16", "materialized"):
        meta = {kk: v[kk] for kk in v if kk not in ("assignment", "config")}
        c[(a.get("backend"), a.get("dynamic_tile_m"), a.get("nvfp4_share_input"), a.get("nvfp4_materialize_intermediate"), json.dumps(meta)[:200])] += 1
for k, n in sorted(c.items(), key=lambda x: str(x)): print(n, k)

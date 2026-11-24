import re, sys
for name in sys.argv[1:]:
    L = re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", open(f"/tmp/b1fence/sass/{name}.sass").read())
    res = []
    prev = 0
    for i, (a, o) in enumerate(L):
        if "SYNCS.ARRIVE.TRANS64.A1T0" not in o:
            continue
        seg = L[prev:i]; prev = i + 1
        last_ld = max([k for k, (_, x) in enumerate(seg) if x.startswith("LDSM") or x.startswith("LDS")], default=-1)
        mb = [k for k, (_, x) in enumerate(seg) if "MEMBAR" in x]
        fv = [k for k, (_, x) in enumerate(seg) if "FENCE.VIEW.ASYNC.S" in x]
        ok = bool(mb) and bool(fv) and mb[-1] > last_ld and fv[-1] > mb[-1]
        res.append((a, ok, len(seg) - 1 - (fv[-1] if fv else -1)))
    print(name, "consumer releases:", len(res), " fence after last LDSM of the k-block:", sum(r[1] for r in res),
          " instrs fence->arrive min/max:", min(r[2] for r in res), max(r[2] for r in res))

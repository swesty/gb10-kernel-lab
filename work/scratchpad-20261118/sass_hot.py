import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]; data = rows[2:]
iA, iS, iW = hdr.index("Address"), hdr.index("Source"), hdr.index("Warp Stall Sampling (All Samples)")
iE = hdr.index("Instructions Executed")
recs = []
for r in data:
    try: recs.append((int(r[iW] or 0), int(r[iA], 16) if r[iA].startswith("0x") else int(r[iA]), r[iS], int(r[iE] or 0)))
    except ValueError: pass
recs.sort(key=lambda x: x[1])
tot = sum(r[0] for r in recs)
print("total samples", tot, "instructions", len(recs))
idx = {r[1]: i for i, r in enumerate(recs)}
for w, a, s, e in sorted(recs, reverse=True)[:18]:
    i = idx[a]
    ctx = " || ".join(x[2][:60] for x in recs[max(0, i - 3):i])
    print(f"{100*w/tot:5.1f}%  {a:#07x} x{e:<9d} {s[:70]:70s}  <- {ctx}")

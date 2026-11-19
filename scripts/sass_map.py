"""Coarse code map of an ncu SASS export: samples per region with landmark opcodes."""
import csv, sys, re, collections
rows = list(csv.reader(open(sys.argv[1])))
hdr = rows[1]; data = rows[2:]
iA, iS, iW, iE = (hdr.index(k) for k in ("Address", "Source", "Warp Stall Sampling (All Samples)", "Instructions Executed"))
recs = []
for r in data:
    try: recs.append((int(r[iA], 16), r[iS].strip(), int(r[iW] or 0), int(r[iE] or 0)))
    except ValueError: pass
recs.sort()
tot = sum(r[2] for r in recs)
def op(s):
    s = re.sub(r"^@!?U?P\w+\s+", "", s)
    return s.split()[0] if s else ""
marks = {"MMA": ("OMMA", "QMMA", "HMMA", "UTCQMMA", "MMA"), "TMA": ("UTMALDG", "UBLKCP", "UTMACCTL"),
         "RED": ("RED", "ATOM", "ATOMG"), "BAR": ("BAR",), "WAIT": ("SYNCS",), "STS": ("STS", "STSM"),
         "LDG": ("LDG",), "STG": ("STG",), "LDSM": ("LDSM",)}
B = int(sys.argv[2]) if len(sys.argv) > 2 else 96
print(f"total samples {tot}, {len(recs)} instructions, region = {B} instructions")
for i in range(0, len(recs), B):
    chunk = recs[i:i + B]
    w = sum(r[2] for r in chunk)
    cnt = collections.Counter()
    for a, s, ww, e in chunk:
        o = op(s)
        for k, prefixes in marks.items():
            if any(o.startswith(p) for p in prefixes):
                cnt[k] += 1
    ex = max(r[3] for r in chunk)
    if w / tot < 0.004 and not cnt.get("MMA") and not cnt.get("RED"):
        continue
    tag = " ".join(f"{k}{v}" for k, v in sorted(cnt.items()))
    print(f"{chunk[0][0] & 0xfffff:#07x} {100*w/tot:5.1f}%  maxexec {ex:>8d}  {tag}")

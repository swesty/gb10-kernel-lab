"""For every MTPPrefill entry in a b12x compile cache: extract the fatbin, disassemble, and check that each
consumer release (plain SYNCS.ARRIVE.TRANS64.A1T0) has MEMBAR then FENCE.VIEW.ASYNC.S after the last LDS/LDSM of its k-block."""
import glob, json, os, re, struct, subprocess, sys
root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for js in sorted(glob.glob(root + "/**/*.json", recursive=True)):
    txt = open(js).read()
    if "MTPPrefill" not in txt and "mtp_feedback" not in txt:
        continue
    o_path = js[:-5] + ".o"
    if not os.path.exists(o_path):
        continue
    blob = open(o_path, "rb").read(); o = blob.find(bytes.fromhex("50ed55ba"))
    _, _, hs, fs = struct.unpack_from("<IHHQ", blob, o)
    name = os.path.basename(js)[:12]
    fb = f"{out}/{name}.fatbin"; open(fb, "wb").write(blob[o:o + hs + fs])
    sass = subprocess.run(["cuobjdump", "-sass", fb], capture_output=True, text=True).stdout
    open(f"{out}/{name}.sass", "w").write(sass)
    L = re.findall(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", sass)
    res, prev = [], 0
    for i, (a, op) in enumerate(L):
        if "SYNCS.ARRIVE.TRANS64.A1T0" not in op:
            continue
        seg = L[prev:i]; prev = i + 1
        last_ld = max([k for k, (_, x) in enumerate(seg) if x.startswith("LDS")], default=-1)
        mb = [k for k, (_, x) in enumerate(seg) if "MEMBAR" in x]
        fv = [k for k, (_, x) in enumerate(seg) if "FENCE.VIEW.ASYNC.S" in x]
        res.append(bool(mb) and bool(fv) and mb[-1] > last_ld and fv[-1] > mb[-1])
    spec = re.findall(r"\(\d+, \d+, \d+[^)]*\)", txt)[:1]
    print(f"{name} instrs={len(L)} consumer_releases={len(res)} fenced_after_last_load={sum(res)} spec={spec}")

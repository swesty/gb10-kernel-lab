import json, glob, re, subprocess, os, sys
root=sys.argv[1]
targets=dict(a.split("=",1) for a in sys.argv[2:])
ins_re=re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")
def regs(s): return set(int(x) for x in re.findall(r"\bR(\d+)\b",s))
def dest_range(op):
    m=re.match(r"(?:@!?U?P\d+\s+)?(LDSM\S*|LDS\S*)\s+R(\d+)",op)
    if not m: return None
    name,r=m.group(1),int(m.group(2))
    if name.startswith("LDSM"):
        n=int(name.rsplit(".",1)[-1]) if name.rsplit(".",1)[-1].isdigit() else 1
    else:
        n={"128":4,"64":2}.get(name.rsplit(".",1)[-1],1)
    return set(range(r,r+n))
os.makedirs("/tmp/b1fence/sass",exist_ok=True)
for name,path in targets.items():
    import struct
    blob=open(path,"rb").read(); o=blob.find(bytes.fromhex("50ed55ba"))
    magic,ver,hs,fs=struct.unpack_from("<IHHQ",blob,o)
    fb="/tmp/b1fence/sass/%s.fatbin"%name; open(fb,"wb").write(blob[o:o+hs+fs])
    sass=subprocess.run(["cuobjdump","-sass",fb],capture_output=True,text=True).stdout
    open(f"/tmp/b1fence/sass/{name}.sass","w").write(sass)
    lines=[(a,o) for a,o in ins_re.findall(sass)]
    arch=re.findall(r"arch = (\S+)",sass)[:1]
    funcs=re.findall(r"Function : (\S+)",sass)
    print(f"\n######## {name}  {os.path.basename(path)[:12]}  arch={arch} funcs={len(funcs)} instrs={len(lines)}")
    print("  LDSM:",sum('LDSM' in o for a,o in lines)," HMMA:",sum('HMMA' in o for a,o in lines),
          " SYNCS.ARRIVE:",sum('SYNCS.ARRIVE' in o for a,o in lines)," UTMALDG:",sum('UTMALDG' in o for a,o in lines),
          " FENCE/MEMBAR:",sum(('FENCE' in o or 'MEMBAR' in o) for a,o in lines))
    for i,(a,o) in enumerate(lines):
        if "SYNCS.ARRIVE" not in o: continue
        # look back up to 60 instrs for stage loads
        loads=[]; 
        for k in range(i-1,max(-1,i-60),-1):
            ok=lines[k][1]
            if "SYNCS" in ok or "BRA" in ok or "EXIT" in ok: break
            dr=dest_range(ok)
            if dr: loads.append((k,dr))
        unconsumed=0; fenced=any(('FENCE' in lines[k][1] or 'MEMBAR' in lines[k][1]) for k in range(max(0,i-4),i))
        for k,dr in loads:
            used=any(regs(",".join(lines[m][1].split(",")[1:]) if not lines[m][1].startswith("HMMA") else lines[m][1].split(" ",1)[1]) & dr
                     for m in range(k+1,i) if not dest_range(lines[m][1]) or True)
            # proper: any instruction between k and i that reads dr as a source
            used=False
            for m in range(k+1,i):
                op=lines[m][1]
                parts=op.split(None,1)
                if len(parts)<2: continue
                operands=parts[1]
                if parts[0].startswith("@"):  # predicated
                    sub=operands.split(None,1); operands=sub[1] if len(sub)>1 else ""
                srcs=",".join(operands.split(",")[1:])
                if regs(srcs)&dr: used=True; break
            if not used: unconsumed+=1
        # what comes right after the arrive
        nxt=[lines[m][1].split()[0] if not lines[m][1].startswith("@") else lines[m][1].split()[1] for m in range(i+1,min(len(lines),i+4))]
        kind="tx" if not o.rstrip().endswith("RZ") else "plain"
        print(f"  arrive @{a} [{kind}] loads_in_window={len(loads)} unconsumed={unconsumed} fence_before={fenced} next={nxt}")
        if unconsumed and name.startswith("B1"):
            for m in range(max(0,i-4),min(len(lines),i+3)): print(f"      /*{lines[m][0]}*/ {lines[m][1]}")

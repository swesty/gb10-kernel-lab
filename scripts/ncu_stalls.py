import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr, units, vals = rows[0], rows[1], rows[2]
st = []
for h, u, v in zip(hdr, units, vals):
    if h.startswith("smsp__pcsamp_warps_issue_stalled_") and not h.endswith("not_issued"):
        try: st.append((float(v.replace(",", "")), h))
        except ValueError: pass
tot = sum(v for v, h in st)
print("pc samples total", tot)
for v, h in sorted(st, reverse=True)[:14]:
    print(f"{v:12.0f} {100 * v / tot:6.1f}%  {h.replace('smsp__pcsamp_warps_issue_stalled_', '')}")
for key in ("dram__bytes_read.sum", "lts__t_sectors_op_red.sum", "lts__t_sectors_op_atom.sum",
            "smsp__inst_executed_op_global_red.sum", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__cycles_elapsed.avg", "lts__t_bytes.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"):
    if key in hdr: print(f"{key:60s} {vals[hdr.index(key)]:>18s} {units[hdr.index(key)]}")

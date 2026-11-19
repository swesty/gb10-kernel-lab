"""Print a coarse timeline from an ncu PM-sampling report (one kernel)."""
import sys
sys.path.insert(0, "/opt/nvidia/nsight-compute/2025.3.1/extras/python")
import ncu_report
ctx = ncu_report.load_report(sys.argv[1])
act = ctx.range_by_idx(0).action_by_idx(0)
names = [n for n in act.metric_names() if n.startswith("pmsampling:")]
want = ["pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
        "pmsampling:sm__inst_executed_realtime.avg.per_cycle_active",
        "pmsampling:lts__average_t_sector_hit_rate_realtime.pct"]
want += [n for n in names if "dram" in n or "lts__t_sectors" in n or "throughput" in n][:3]
want += [n for n in names if "stalled_sleeping" in n or "stalled_barrier" in n or "stalled_long_scoreboard" in n or "stalled_membar" in n or "stalled_branch" in n]
series = {}
for n in want:
    m = act.metric_by_name(n)
    if m is None: continue
    k = m.num_instances()
    vals = [m.as_double(i) for i in range(k)]
    try:
        cor = m.correlation_ids(); ts = [cor.as_uint64(i) for i in range(k)]
    except Exception:
        ts = list(range(k))
    series[n] = (ts, vals)
if not series:
    print("no series; names:", names[:40]); sys.exit()
nb = 25
print("samples per series:", {n.split(':')[1][:40]: len(v[1]) for n, v in series.items()})
print("t(%)  " + "  ".join(f"{n.split(':')[1].split('.')[0][-22:]:>22s}" for n in series))
for b in range(nb):
    row = []
    for n, (ts, vals) in series.items():
        t0, span = ts[0], (ts[-1] - ts[0]) or 1
        lo, hi = t0 + span * b / nb, t0 + span * (b + 1) / nb
        sel = [v for t, v in zip(ts, vals) if lo <= t < hi] or [float("nan")]
        row.append(sum(sel) / len(sel))
    print(f"{100*b/nb:4.0f}  " + "  ".join(f"{v:22.2f}" for v in row))

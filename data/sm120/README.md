# SM120 cross-check (RTX PRO 6000, 2026-11-19)

Re-runs of the b12x findings on an RTX PRO 6000 Blackwell Workstation (sm_120, 188 SMs, 128 MB L2), with the same
b12x 1.5.0 source and harnesses as on GB10. Write-up: REPORT §13.21; driver: `scripts/sm120/all.sh` (via `dr.sh`).

| Files | What |
|---|---|
| `driver.log`, `outage_start`, `outage_end`, `gpu_state.txt` | Run log (production model stopped 17:36:30–17:44:08 UTC), clocks and power |
| `mma_peak.txt` | `mma.sync` tensor-core peaks |
| `mtp_stock_f*.txt`, `mtp_exp_<tileN>_<stages>_<rasterN>_f<flushMB>.txt` | MTP feedback prefill GEMM variants (`f0` back-to-back, `f512` after a 512 MB L2 flush) |
| `dense_default.json/.log` | Dense MXFP8 live path (BlockscaledQuery) and the default lowering sm_120 picks |
| `dense_sweep_stock.json`, `dense_sweep_rm=0,sw=1.json` | Dense tile configs with the stock scheduler and with N-major (summary: `scripts/sm120/dense_summary.py`) |
| `attn_*_c<context>.txt` | Selected-position attention timings per variant |
| `attn-{stock,swz}-sm120.ncu-rep` (+ `-details.csv`, `-raw.csv`) | Nsight Compute, stock and `S<3,3,3>`-only |
| `ncu_m_{pipe,pipeswz,all}.csv` | Shared-memory counters for the pipelined variants (`scripts/sm120/ncu_pipe_variants.sh`) |
| `gb10-same-counters/gb10_*.csv` | The same counters on GB10 (spark-5, beside the live engine), for a like-for-like comparison |

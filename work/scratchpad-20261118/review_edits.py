"""Exact-string edits to REPORT.md for the accuracy review. Each old string must match exactly once."""
import sys

P = "/Users/swesty/Developer/projects/gb10-kernel-lab/REPORT.md"
s = open(P).read()

E = []

# --- title / headline -------------------------------------------------------------------------
E.append(("profiling and kernel work, 2026-11-18\n", "profiling and kernel work, 2026-11-18/19\n"))
E.append((
    "| Tried and rejected | per-row TMA bulk-copy gather (0.54–1.12×); multi-row \"union\" attention (0.87–1.01×); `torch.compile`; MoE work-source/tile configs; MXFP8 HC weights (decode +2–3%, prefill −7%); 8K prefill chunks (−25% KV cache); b12x exhaustive autotune (host OOM) |",
    "| Tried and rejected | per-row TMA bulk-copy gather (0.54–1.12×); multi-row \"union\" attention (0.87–1.01×); `torch.compile`; MoE work-source and tile configs, incl. tile 128 (prefill 6% slower); deterministic MoE output (1.7–2.5× slower in the harness, does not start in the engine); deeper MoE pipeline and barrier-free hand-offs (no gain); MXFP8 HC weights (decode +2–3%, prefill 7% slower); 8K prefill chunks (−25% KV cache); b12x exhaustive autotune (host OOM); FlashInfer 0.7.5 (nothing applicable to GB10) |\n"
    "| Investigated and stopped | an MoE kernel rewrite: per-kernel counters put the prefill MoE at 75% of the bandwidth a streaming read reaches on this GB10, leaving ~3–4% of prefill *[estimate]* (§13.11) |",
))

# --- §2 hardware --------------------------------------------------------------------------------
E.append((
    "| Memory bandwidth | 273 GB/s | spec; the highest live sample read exactly 273 GB/s |",
    "| Memory bandwidth | 273 GB/s theoretical; **236–238 GB/s** reached by a plain streaming read, 225 GB/s read+write | spec; `bw.py` (§13.11). One 50 µs live sample of the L2-miss proxy read 273 GB/s (§7) |",
))
E.append((
    "| 273 GB/s | Decode is bandwidth-bound (69% of peak on average, big kernels 70–82%); there is little decode headroom in the weight-streaming kernels. |",
    "| 273 GB/s (236–238 GB/s reachable) | Decode is bandwidth-bound: 69% of the 273 GB/s spec on average (79% of what a streaming read reaches), big kernels 70–82% of spec; there is little decode headroom in the weight-streaming kernels. The prefill MoE turned out to be close to this limit too (§13.11). |",
))

# --- §4 triage ------------------------------------------------------------------------------------
old_flow_start = "```mermaid\nflowchart TD\n  A[Live metrics: what does traffic look like?]"
i = s.index(old_flow_start)
j = s.index("```\n", i + 10) + 4
new_flow = '''```mermaid
flowchart TD
  A[Live metrics: what does traffic look like?] --> B[Nsight Systems on the live engine: share of GPU time per kernel]
  B --> C{Is GPU idle between kernels?}
  C -- no --> D[Live GPU metric sampling: bandwidth / SM / tensor activity per kernel]
  D --> E{Kernel near a hardware limit?}
  E -- "yes (decode MoE, decode dense GEMM)" --> F[Little headroom: deprioritise]
  E -- "no (attention, combine-norm, HC GEMMs; prefill MoE read as 40%)" --> G[Nsight Compute: occupancy, stalls, conflicts]
  G --> H{Can ncu run on the live engine?}
  H -- "no: crashed (cuBLAS in CUDA graph)" --> I[Side engine: 4-layer model, eager, engine stopped]
  I --> J[Target 1: prefill attention, low tensor use]
  J --> K[Standalone harness at live shapes + FP32 reference]
  K --> L{Harness matches live per-row cost?}
  L -- "yes (3.4 vs 3.85 us/row/layer)" --> M[Read source + profile: root causes]
  M --> N[Fix one cause at a time, re-measure, keep only wins]
  N --> O[Live A/B + logprob check against the engine's own noise floor]
  M --> Q[Structural alternative: multi-row union kernel]
  Q --> R[Measure real selection overlap from live dumps]
  R --> S[Cheap prototype with existing kernel] --> T{Wins?}
  T -- "no (0.87-1.01x)" --> U[Reject]
  O --> V[Target 2: configuration levers: torch.compile, MoE knobs, chunk size, autotune]
  V --> W{Win without a cost?}
  W -- no --> U
  O --> X[Target 3: HC chain and PyTorch glue: trace each kernel back to its source line]
  X --> Y{Fusible, or an identity case doing wasted work?}
  Y -- yes --> Z[Fuse or skip, A/B, logprob check]
  Z --> AA[Re-trace the current build: category table]
  AA --> AB[Target 4: prefill MoE, now the largest item]
  AB --> AC[Harness on captured live routes reproduces live time]
  AC --> AD[ncu stalls + PM timeline: TMA pipeline waits]
  AD --> AE[Cheap upper-bound tests: more stages, no hand-off barriers]
  AE --> AF{Faster?}
  AF -- no --> AG[Per-kernel bytes vs measured achievable bandwidth]
  AG --> AH{Headroom worth a rewrite?}
  AH -- "no (75% of achievable)" --> AI[Stop]
```
'''
s = s[:i] + new_flow + s[j:]

E.append((
    "| 7 | Is a bigger redesign worth it? | Live selection dumps + prototype | Overlap decides the design | Rejected |",
    "| 7 | Is a bigger redesign worth it? | Live selection dumps + prototype | Overlap decides the design | Rejected |\n"
    "| 8 | Do configuration levers help? | Engine flags and env switches via the launcher | No code to write | `torch.compile` unsupported; MoE knobs, 8K chunks and exhaustive autotune rejected (§13.1–13.3, §13.9, §13.10) |\n"
    "| 9 | Can the HyperConnection chain be fused? | Microbenchmarks + Triton kernels | Bytes and launches are the lever | −5% prefill (§13.4) |\n"
    "| 10 | Where does the PyTorch glue come from? | Trace neighbouring kernels back to source | Shares hide per-chunk repetition | GDN identity fast path and fused q/k norm: −9% prefill together (§13.6, §13.7) |\n"
    "| 11 | Is the prefill MoE fixable? | Harness on live routes, Nsight Compute stalls, PM sampling, byte counters, `bw.py` | Size the headroom before a rewrite | 75% of achievable bandwidth; rewrite stopped (§13.11) |",
))

# --- §7 ---------------------------------------------------------------------------------------------
E.append((
    "attached to a dummy `sleep` measured the live engine **without a restart**. The decode peak sample read\nexactly 273 GB/s, which validates the proxy. Then a restart",
    "attached to a dummy `sleep` measured the live engine **without a restart**. The decode peak sample read\n273 GB/s, the spec figure, so the proxy is on the right scale. It is approximate: a plain streaming read\nreaches only 236–238 GB/s (§13.11), so single samples can overshoot, and for the prefill MoE this proxy\nlater disagreed with a per-kernel counter (below). Then a restart",
))
E.append((
    "| Kernel | Share | Memory read | SMs busy | Tensor |",
    "| Kernel | Share | Memory read (% of 273 GB/s) | SMs busy | Tensor |",
))
E.append((
    "| Prefill: MoE | 22.7% | 110 GB/s (40%) | 94% | 12% |",
    "| Prefill: MoE | 22.7% | 110 GB/s (40%)* | 94% | 12% |",
))
E.append((
    "**Decision:** attention in prefill was occupied but neither memory- nor compute-bound",
    "\\* Superseded for the MoE: Nsight Compute's per-kernel fill counter on a standalone harness measured\n178 GB/s, 75% of what a streaming read reaches (§13.11). The two counters differ (L2 read-lookup misses vs\nall L2 fills from memory) and were taken under different conditions (live engine vs harness); the gap is\nnot reconciled.\n\n**Decision:** attention in prefill was occupied but neither memory- nor compute-bound",
))

# --- §8.3 -------------------------------------------------------------------------------------------
E.append((
    "| MoE (decode) | (unreliable under ncu) | — | 7% | 10% / 10% | 182 / 54 KB | sleeping (spin barriers) |",
    "| MoE (decode) | (unreliable under ncu) | — | 7% | 10% / 10% | 182 / 54 KB | sleeping (first read as spin barriers; in prefill traced to TMA-pipeline waits, §13.11) |",
))
E.append((
    "path is conflict-free thanks to padded/XOR-swizzled rows) — but MoE is limited by its spin barriers, not by\nconflicts.",
    "path is conflict-free thanks to padded/XOR-swizzled rows) — but conflicts are not what limits MoE. Its top\nstall, \"sleeping\", turned out to be waits on its TMA pipeline's mbarriers (§13.11), not grid spin barriers\nas first assumed.",
))

# --- §9.3 / §9.4 --------------------------------------------------------------------------------------
E.append((
    "3-slot ring; convert and store after PV(i). Result: 13.94 → 8.60 ms (16K).",
    "3-slot ring; convert and store after PV(i). Result: 13.94 → 8.60 ms (16K; the separate run behind the\n§9.8 table measured 14.04 → 8.68 ms).",
))
E.append((
    "| Real-text decode c=4, temperature 0 | 8.6–9.0 tok/s | 8.6–8.9 tok/s |",
    "| Real-text decode c=4, temperature 0 | 8.6–9.0 chunks/s† | 8.6–8.9 chunks/s† |",
))
E.append((
    "Greedy outputs differ between the arms (both coherent).",
    "† Per request. The decode script at this stage counted streamed chunks, not tokens. With MTP each chunk\ncarries one decode step's tokens, ~3 at the measured acceptance (1 + ~2.05 accepted drafts), so\n8.6–9.0 chunks/s is ~26–28 tokens/s per request. From §13.4 on, decode figures are tokens/s taken from the\nresponse's usage count. Arm-to-arm comparisons within a table are unaffected. §13.1–13.3 use chunks/s too.\n\nGreedy outputs differ between the arms (both coherent).",
))
E.append((
    "noise floor. *[inference]* The likely source is b12x's dynamic MoE kernels (b12x has a\n`B12X_DYNAMIC_DETERMINISTIC_OUTPUT` switch); not confirmed.",
    "noise floor. One source was confirmed later (§13.11): the MoE kernel reduces its split-K partial sums with BF16\natomic adds in whatever order CTAs finish, and two harness runs of the same configuration on the same input\ndiffer by ~6e-3 relative. Its deterministic mode is 1.7–2.5× slower and does not start in the engine.\nWhether other kernels add to the noise was not tested.",
))

# --- §13 table ----------------------------------------------------------------------------------------
E.append((
    "| Decode GDN at 4 requests | 10.3% | 17% bandwidth, latency-bound | rework state update |",
    "| Decode GDN at 4 requests | 10.3% | 17% bandwidth, latency-bound | rework state update (not attempted) |",
))

# --- §13.1–13.3 decode labels -------------------------------------------------------------------------
E.append((
    "| Real-text decode c=4 | 8.6–8.9 tok/s | 8.3–8.8 tok/s |",
    "| Real-text decode c=4 (chunks/s, §9.4) | 8.6–8.9 | 8.3–8.8 |",
))
E.append((
    "| MoE setting | Real-text 14.5K prefill | Random 16K TTFT | Decode c=4 |",
    "| MoE setting | Real-text 14.5K prefill | Random 16K TTFT | Decode c=4 (chunks/s, §9.4) |",
))
E.append(("| **6.23–6.26 s** | **6,705 ms** | 8.6–9.2 tok/s |", "| **6.23–6.26 s** | **6,705 ms** | 8.6–9.2 |"))
E.append(("| 6.31–6.33 s (+1.2%) | 6,803 ms | 8.6–8.9 tok/s |", "| 6.31–6.33 s (+1.2%) | 6,803 ms | 8.6–8.9 |"))
E.append(("| 41,824 ms | 2.6 tok/s (3.4× slower) |", "| 41,824 ms | 2.6 (3.4× slower) |"))
E.append((
    "**Conclusion:** the defaults are the best available work-distribution settings; multi-CTA is essential.",
    "**Conclusion:** the defaults are the best available work-distribution settings; multi-CTA is essential. (The\n\"sleeping\" stall later turned out to be TMA-pipeline waits, not spin barriers; §13.11.)",
))
E.append((
    "| MoE tile | Real-text 14.5K prefill | Random 16K TTFT | Decode c=4 |",
    "| MoE tile | Real-text 14.5K prefill | Random 16K TTFT | Decode c=4 (chunks/s, §9.4) |",
))
E.append((
    "originally mislabelled M = 128) | 6.23–6.26 s | 6,705 ms | 8.6–9.2 tok/s |",
    "originally mislabelled M = 128) | 6.23–6.26 s | 6,705 ms | 8.6–9.2 |",
))
E.append(("| `64x128` | 6.26–6.27 s | 6,737 ms | 8.5–8.9 tok/s |", "| `64x128` | 6.26–6.27 s | 6,737 ms | 8.5–8.9 |"))
E.append((
    "| `32x128` | 6.27–6.28 s | 6,725 ms | 8.7–9.0 tok/s |",
    "| `32x128` (same tile as the default) | 6.27–6.28 s | 6,725 ms | 8.7–9.0 |",
))
E.append((
    "kernel. *[inference]* The candidates left are its inter-phase spin-barrier coordination (its top stall) and\nFP4 unpacking / routing work. Confirming that needs a profile of the MoE kernel under realistic routing, which\nthe dummy-weight side engine cannot provide.",
    "kernel. At the time the candidates were its spin-barrier coordination and FP4 unpacking / routing work, and\nconfirming either needed a profile under realistic routing that the dummy-weight side engine could not give.\nA standalone harness on captured live routes later provided it (§13.11): the top stall is waiting on the TMA\nweight pipeline, the routing front end is ~8% of the kernel, and the kernel moves memory at 75% of what a\nstreaming read reaches.",
))

# --- §13.8 ---------------------------------------------------------------------------------------------
E.append((
    "| Prefill window | 7,074 ms | 5,777 ms | (the current trace predates §13.7; expect ~5,550 ms with it) |",
    "| Prefill window | 7,074 ms | 5,777 ms | trace taken before §13.7; random-16K TTFT fell a further 6,122 → 5,800 ms with §13.7 and v2 |",
))
i = s.index("**The MoE kernel, sized against the hardware.**")
j = s.index("Smaller fusion targets still in the elementwise bucket")
new_moe = '''**The MoE kernel, sized against the hardware.** Per 4,096-token chunk and layer: 512 experts × top-10 =
40,960 token-expert pairs. Each expert has W13 = 2 × 640 × 2,560 and W2 = 2,560 × 640 NVFP4 weights, about
2.8 MB with scales, so roughly 1.4 GB if every expert is touched (1.47 GB with the activations).

| Bound | Per layer-chunk |
|---|---|
| Unique bytes (1.47 GB) at 238 GB/s, what a streaming read reaches (§13.11) | ~6.2 ms (5.4 ms at the 273 GB/s spec) |
| FP4 tensor math, 403 GFLOP | ~1–2 ms *[estimate; GB10 dense FP4 `mma.sync` peak not measured]* |
| `MoEDynamicKernelSilu`, live trace | **9.4–10.0 ms** |
| Same kernel, standalone harness on live routes (§13.11) | 8.95–9.2 ms |

The kernel launches a persistent grid of 48 CTAs (one per SM) of 160 threads, with 182 registers per thread
and 54 KB of shared memory: 5 warps per SM to hide memory latency.

At this point the MoE looked like the largest remaining lever. Live sampling had put it at 40% of bandwidth
(§7), and the 8K-chunk experiment (§13.9) saved much less than a weight-bytes model predicts, which suggested
a latency- or synchronisation-bound kernel with ~0.65 s (~12%) of a 16K prefill to gain *[estimate]*. §13.11
tested that and found otherwise: per-kernel counters put the kernel at 75% of achievable bandwidth, reading
only 8% more than the unique bytes, which leaves about 3–4% of prefill for a large rewrite. Why the 8K-chunk
saving was smaller than a weight-bytes model predicts is not established.

'''
s = s[:i] + new_moe + s[j:]
E.append((
    "- `torch.zeros` of `core_attn_out` before the GDN conv, unnecessary on the identity path: 27 ms;\n- router scaling multiply before `topkGating`: 28 ms;\n- query copy after mRoPE: 20 ms (likely gone with §13.7).\n\nTogether that is about 2% *[estimate]*.",
    "- `torch.zeros` of `core_attn_out` before the GDN conv, unnecessary on the identity path: 27 ms (**done**,\n  §13.7 v2);\n- router scaling multiply before `topkGating`: 28 ms;\n- query copy after mRoPE: 20 ms (the fused kernel of §13.7 writes the query in place, so this should be gone;\n  not re-traced).\n\nThe two open items total ~70 ms, about 1.3% of the current prefill *[estimate]*.",
))

# --- §13.9 ---------------------------------------------------------------------------------------------
E.append((
    "**Why.** Each 4,096-token chunk re-reads every expert's weights (~1 GB per MoE layer *[estimate]*) and all",
    "**Why.** Each 4,096-token chunk re-reads every expert's weights (~1.4 GB per MoE layer; 1.59 GB of L2 fills\nmeasured later, §13.11) and all",
))
E.append((
    "| | 4,096 (production) | 8,192 |",
    "| | 4,096 (production, build as of §13.7) | 8,192 |",
))

# --- §13.11 MoE ----------------------------------------------------------------------------------------
E.append((
    "The target is `MoEDynamicKernelSilu`: 27% of prefill and 9.4–10 ms per 4,096-token layer-chunk, against a\n~5 ms weight-streaming floor (§13.8).",
    "The target is `MoEDynamicKernelSilu`: 27% of prefill and 9.4–10 ms per 4,096-token layer-chunk. When this\nsection started, the working estimate was a ~5 ms weight-streaming floor at the 273 GB/s spec; the measured\nfloor is ~6.2 ms (§13.8 table).",
))
E.append((
    "Things that look costly for a large prefill *[inference from the code, not yet measured]*:",
    "Things that looked costly for a large prefill *[inference from the code; each was checked later in this\nsection, outcome in italics]*:",
))
E.append((
    "   (order-dependent rounding).\n2.",
    "   (order-dependent rounding). *Measured: the L2 atomic unit is 10% busy and atomic throttling is 0.1% of\n   stalls. Not a limit, but it is the source of the run-to-run differences (§9.4).*\n2.",
))
E.append((
    "   with the 12% tensor-pipe activity measured earlier.\n3.",
    "   with the 12% tensor-pipe activity measured earlier. *Measured: 4 stages gave no gain. Not the limit.*\n3.",
))
E.append((
    "   close together).\n4.",
    "   close together). *Measured: TMA reads 4.72 GB per call from L2, but only 1.59 GB reaches memory; L2\n   absorbs the re-reads.*\n4.",
))
E.append((
    "4. **Serial front end:** 5 grid barriers, two rounds of 40,960 global atomics, a single-warp scan.",
    "4. **Serial front end:** 5 grid barriers, two rounds of 40,960 global atomics, a single-warp scan.\n   *Measured: ~8% of the kernel (PM timeline).*",
))
E.append((
    "   (`mbarrier.try_wait`), not from the grid barrier. The grid barrier spins on a single thread without\n   nanosleep.",
    "   (`mbarrier.try_wait`), not from the grid barrier. The grid barrier spins on a single thread without\n   nanosleep. *Confirmed: Nsight Compute places the \"sleeping\" samples on those mbarrier waits.*",
))
i = s.index("Open question: which tile M the live kernel runs.")
j = s.index("**Live config tests:**")
new_tile = '''**Which tile M runs live.** The launch's 54,272 B of shared memory matches the M = 32 stage sizing, and
b12x's selection cache (`~/.cache/b12x/compile/preparation/*.json`) shows the tuner choosing dynamic tile
M ∈ {16, 32, 64} with shared input for this MoE. So the "default (M = 128)" label first used in §13.3 was
wrong: that experiment compared 32 and 64 against an M = 32 default.

'''
s = s[:i] + new_tile + s[j:]
E.append((
    "**Diagnosis.** The compute loop is starved for data, and the cause is specific to GB10.",
    "**First diagnosis (tested and disproven below).** The compute loop is starved for data, and the cause is\nspecific to GB10.",
))
E.append((
    "- At tile 32 each task streams ~360 KB of FC1 tiles and 160 KB of FC2 tiles: about 3.8 GB per call\n  *[estimate]*, mostly L2 hits,",
    "- At tile 32 each task streams ~360 KB of FC1 tiles and 160 KB of FC2 tiles: about 3.8 GB per call\n  *[estimate; 4.72 GB measured below]*, mostly L2 hits,",
))
E.append((
    "| Option | Targets | Expected | Risk |\n|---|---|---|---|",
    "| Option | Targets | Expected | Risk | Outcome |\n|---|---|---|---|---|",
))
E.append((
    "| largest *[estimate]* | layouts and TMA descriptors in a 9,800-line CuTe DSL kernel |",
    "| largest *[estimate]* | layouts and TMA descriptors in a 9,800-line CuTe DSL kernel | 4 stages tested: no gain |",
))
E.append((
    "| moderate; atomics are not the top stall (lg_throttle 0.1%) | medium |",
    "| moderate; atomics are not the top stall (lg_throttle 0.1%) | medium | not pursued: the no-hand-off bound below caps its pipelining gain |",
))
E.append((
    "| weight re-streaming for hot experts | uncertain | needs two launches over partitioned routes |",
    "| weight re-streaming for hot experts | uncertain | needs two launches over partitioned routes | not pursued: L2 already absorbs the re-reads |",
))
E.append((
    "- The floor is ~6.2 ms against 8.95 ms measured, so the theoretical headroom is ~30% of the kernel. Pipeline\n  depth and producer/consumer hand-offs are ruled out as the limit.",
    "- The floor is ~6.2 ms against 8.95 ms measured, so the theoretical headroom is ~30% of the kernel (~40% if\n  the full 273 GB/s spec were reachable; no sustained measurement here exceeded 238 GB/s). Pipeline depth and\n  producer/consumer hand-offs are ruled out as the limit.",
))
E.append((
    "- The earlier \"~12% of prefill\" estimate (§13.8) assumed the kernel was at 40% of bandwidth. That figure came\n  from live GPU-metric sampling shared with neighbouring kernels; the per-kernel counters here supersede it.",
    "- The earlier \"~12% of prefill\" estimate (§13.8) assumed the kernel was at 40% of bandwidth. That figure came\n  from the live L2-miss sampling of §7, a different counter taken under different conditions. The per-kernel\n  fill counter is the more direct measurement and is used here; the gap between the two is not explained.",
))

# --- §14 -------------------------------------------------------------------------------------------------
E.append((
    "- Outage windows are recorded in `homelab-build/RUNBOOK.md` (§Spark-5).",
    "- **Device-wide sampling vs per-kernel counters:** live L2-miss sampling put the prefill MoE at 40% of\n  bandwidth; a per-kernel fill counter on a harness put it at 75% of what a streaming read reaches. A kernel\n  plan sized on the first number (§13.8) was dropped after the second (§13.11). Before sizing a rewrite,\n  measure the box's achievable bandwidth and use per-kernel counters.\n"
    "- **Decode metric:** the first decode script counted streamed chunks, which with MTP carry ~3 tokens each.\n  Early c=4 figures (§9.4, §13.1–13.3) are chunks/s.\n"
    "- **b12x harness setup:** the legacy planning API needs the internal weight plan (`experts._impl.plan`), an\n  explicit `cuda:0` device, and `immutable_input_scales=True` to match serving's static input scales.\n"
    "- **Root-owned compile caches** left by `docker exec` experiments have to be removed from inside the\n  container.\n"
    "- Outage windows up to 19:36 UTC on 2026-11-18 are recorded in `homelab-build/RUNBOOK.md` (§Flash Next\n  profiling restarts). The later A/B restarts and the 01:33 UTC OOM on 2026-11-19 are not recorded there yet.",
))

# --- §15 -------------------------------------------------------------------------------------------------
E.append((
    "| Analysis scripts | `attrib.py`, `qsa_overlap.py`, `pair_proto.py`, `ab_real.py`, `lp_check.py` |",
    "| Analysis scripts (`attrib.py`, `qsa_overlap.py`, `pair_proto.py`, `ab_real.py`, `lp_check.py`, `lp_compare.py`) | `gb10-kernel-lab/scripts/` |",
))
E.append((
    "| Current-build prefill capture | `gb10-kernel-lab/data/nsys/D-prefill-16k-best.nsys-rep` |",
    "| Prefill capture of the build through §13.6 (before §13.7) | `gb10-kernel-lab/data/nsys/D-prefill-16k-best.nsys-rep` |",
))
E.append((
    "| MoE harness, Nsight Compute reports (full, PM sampling), timeline/stall scripts | `scripts/moe_harness.py`; spark-5 `~/.cache/vllm/nsys-flashnext/moe-harness-t32*.ncu-rep`; `pm_timeline.py`, `ncu_stalls.py`, `sass_hot.py` |",
    "| MoE harness and analysis scripts | `gb10-kernel-lab/scripts/`: `moe_harness.py`, `pm_timeline.py`, `ncu_stalls.py`, `sass_hot.py`, `sass_map.py`, `selcache.py`, `bw.py`, `stage_sweep.sh`, `stage_sweep2.sh` |\n"
    "| MoE Nsight Compute reports (full, PM sampling) | `gb10-kernel-lab/data/ncu/moe-harness-t32.ncu-rep`, `moe-harness-t32-pm.ncu-rep` |\n"
    "| b12x MoE experiment patches (stage override; unsafe no-hand-off build, timing only) | `gb10-kernel-lab/patches/b12x/` |",
))

bad = []
for old, new in E:
    n = s.count(old)
    if n != 1:
        bad.append((n, old[:90]))
        continue
    s = s.replace(old, new)
if bad:
    for n, o in bad:
        print(f"MATCH {n}: {o!r}")
    sys.exit(1)
open(P, "w").write(s)
print(f"applied {len(E)} edits + 3 block replacements")

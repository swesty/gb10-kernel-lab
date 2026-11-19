"""Second batch: independent-review findings, verified against data."""
import sys

P = "/Users/swesty/Developer/projects/gb10-kernel-lab/REPORT.md"
s = open(P).read()
E = []

# prefill share of request time (§4 table, §5)
E.append(("| Prompt-heavy (31:1), prefill is ~half of request time |",
          "| Prompt-heavy (31:1); prefill is ~26% of request time, ~42% with queueing |"))
E.append(("**Conclusion:** half of each request's wall time is prefill. Prefill improvements matter as much as decode.",
          "**Conclusion:** time to first token is ~42% of a request's wall time (4.56 of ~10.8 s) and prefill compute\n"
          "alone ~26% (2.78 s). Decode is the larger share, but it is bandwidth-bound with little headroom (§7), so\n"
          "prefill is where kernel work can pay off."))
E.append(("~4.2 tokens per step |", "~4.3 tokens per step (the A/B benchmark prompt at temperature 0 accepts 49–58%; acceptance depends on the workload) |"))

# MoE per-launch time and denominators
E.append(("| `MoEDynamicKernelSilu`, live trace | **9.4–10.0 ms** |",
          "| `MoEDynamicKernelSilu`, live trace, per launch (as first recorded) | **9.4–10.0 ms** |\n"
          "| Same, averaged over the random-16K trace (1,366 ms / 196 launches) | ~7.0 ms (random tokens route more evenly, closer to b12x's `disjoint` pattern) |"))
E.append(("| MoE (b12x) | 1,405 ms | 1,366 ms | 27% of what remains — largest single item |",
          "| MoE (b12x) | 1,405 ms | 1,366 ms | 24% of the window (27% of summed kernel time); largest single item |"))
E.append(("| Prefill MoE | 22.7% (27% after other fixes) |",
          "| Prefill MoE | 22.7% (24% of the window after other fixes; 27% of summed kernel time) |"))
E.append(("The target is `MoEDynamicKernelSilu`: 27% of prefill and 9.4–10 ms per 4,096-token layer-chunk. When this",
          "The target is `MoEDynamicKernelSilu`: 24–27% of a random-16K prefill (window vs summed kernel time), ~7 ms\n"
          "per 4,096-token layer-chunk on random prompts and 9.2 ms on captured real-text routing (harness, below;\n"
          "9.4–10 ms as first recorded live). When this"))
E.append(("only 8% more than the unique bytes, which leaves about 3–4% of prefill for a large rewrite.",
          "only 8% more than the unique bytes, which leaves about 2–5% of prefill for a large rewrite, depending on\nrouting."))
E.append(("- Together that is perhaps 10–15% of the kernel, about **3–4% of prefill** *[estimate]*, for a substantial\n  CuTe DSL rewrite.",
          "- Together that is perhaps 10–15% of the kernel. That is about **3–5% of a real-text prefill** (9.2 ms × 196\n"
          "  launches ≈ 1.8 s of 5.43 s) and 2.5–3.5% of a random-16K prefill (1.37 s of 5.8 s) *[estimate]*, for a\n  substantial CuTe DSL rewrite."))
E.append(("leaving ~3–4% of prefill *[estimate]* (§13.11) |", "leaving ~2–5% of prefill *[estimate]* depending on routing (§13.11) |"))

# stage size
E.append(("- Only 2 TMA stages fit. The A buffer is sized for a 128-row atom even at tile 32, and FC1 keeps two B buffers\n"
          "  (gate and up), so a stage is ~27 KB, and only ~99 KB of shared memory is available per SM.\n"
          "- That leaves ~54 KB in flight per SM. On parts with 228 KB the same kernel could keep ~4× more in flight.",
          "- b12x's stage rule picks 2 TMA stages (a shared-memory fit estimate plus a divisor-of-32 rule). The A buffer\n"
          "  is sized for a 128-row atom even at tile 32, and FC1 keeps two B buffers (gate and up). *[Measured below: a\n"
          "  stage is ~21.5 KB, and 4 stages do fit, in 97,280 B.]*\n"
          "- That leaves ~54 KB in flight per SM; parts with 228 KB of shared memory could keep several times more."))

# section refs
E.append(("Swizzle correctness for `ldmatrix` and 128-bit stores matters (§9.6).",
          "Swizzle correctness for `ldmatrix` and 128-bit stores matters (§9.7)."))
E.append(("(cause found later, §9.6)", "(cause found later, §9.7)"))

# §6 vs §7 shares
E.append(("not reconciled.\n\n**Decision:** attention in prefill",
          "not reconciled.\n\nShares in this table come from the GPU-metrics capture (a separate restart) and cover single kernels, so\n"
          "they differ from the §6 categories. §6's \"full attention\" also contains QSA scoring and other\n"
          "attention-side kernels *[inference]*; the paged attention kernel alone is 10.5–10.9% of prefill (here and\n§13.8).\n\n**Decision:** attention in prefill"))

# swizzle "exactly 2x"
E.append(("**exactly 2× the ideal wavefronts regardless of thread mapping**",
          "**about 2× the ideal wavefronts (43–54% excess, §8.3) regardless of thread mapping**"))

# GDN estimate wording and category name
E.append(("about 2.1 ms in all. The first estimate (~1%) counted\none chunk. **The engine",
          "about 2.1 ms in all. **The engine"))
E.append(("one chunk and understate both; see §13.6 for why.",
          "one chunk. The GDN share matches the later measurement (243 ms, 3.4%, §13.8); the q/k norm share\n"
          "understates it (~250 ms, ~3.5%, §13.7)."))
E.append(("the \"GDN gather/scatter/fill\" category\nwent from 243 ms to 0.",
          "the \"GDN gather/scatter\" category\n(the gathers and the scatter kernel) went from 243 ms to 0. The zero fill is counted under elementwise and\nwent later (§13.7 v2)."))
E.append(("| GDN gather/scatter/fill | 243 ms | 0 | §13.6 |", "| GDN gather/scatter | 243 ms | 0 | §13.6 |"))
E.append(("The first GDN estimate missed this by about 3×.",
          "Multiply per-layer-chunk costs by the chunk count before sizing a change."))

# bandwidth ranges
E.append(("big kernels 70–82% of spec;", "the two largest kernels 73–79% of spec (§7);"))
E.append(("| Decode kernels 70–82% of bandwidth;", "| Largest decode kernels 73–79% of bandwidth;"))
E.append(("cuBLAS already 67–84% of bandwidth in isolation", "cuBLAS already 68–84% of bandwidth in isolation"))
E.append(("In isolation cuBLAS already reaches 67–84% of bandwidth", "In isolation cuBLAS already reaches 68–84% of bandwidth"))

# equivalence (recomputed with lp_compare.py against captures A, A2, A3 = original kernel)
E.append(("| Equivalence | every adopted change: logit differences equal the original's own run-to-run noise (§9.4, §13.4, §13.6, §13.7) |",
          "| Equivalence | every adopted change: median log-prob difference 0.0004–0.0005, the same as the original against itself; top-1 agreement and p99 inside, or slightly outside, the range of three original-vs-original pairs (§13.13) |"))
E.append(("| Original (restart) vs patched | 96.56% | 0.0004 / 0.094 / 0.46 | — |",
          "| Original (restart) vs patched | 96.56% | 0.0004 / 0.094 / 0.46 | step 0 |"))
E.append(("Logits within\n  noise (96.60% top-1 agreement, p99 0.457).",
          "Logits within\n  noise (96.54–96.77% top-1 agreement, p99 0.44–0.45; §13.13)."))
E.append(("unchanged, logits within noise (96.69%, p99 0.45).",
          "unchanged, logits within noise (96.67–96.85%, p99 0.42–0.48; §13.13)."))
E.append(("Logits: 96.47% top-1 agreement (control pairs 96.56–96.76%), p99 0.454 (control 0.44–0.46) — p99 inside the\n"
          "noise range, top-1 agreement marginally below the four control samples; treated as noise but flagged.",
          "Logits: 96.46–96.62% top-1 agreement (original-vs-original pairs 96.72–96.76%), p99 0.43–0.47 (pairs\n"
          "0.44–0.47). p99 is inside the range; top-1 agreement is up to 0.3 points below it. Treated as noise, but\nflagged (§13.13)."))
E.append(("| Logits vs earlier captures | — | top-1 agreement 96.62–96.97%, p99 0.43–0.46 (noise floor) |",
          "| Logits vs the three original captures | — | top-1 agreement 96.79–96.97%, p99 0.43–0.46 (noise floor) |"))
E.append(("- Logits: top-1 agreement 96.80–96.90% against three earlier captures, p99 0.45–0.47. That is inside the noise\n  floor,",
          "- Logits: top-1 agreement 96.77–96.89% against the three original captures, p99 0.43–0.45. That is inside the\n  noise floor,"))
E.append(("- Logits: 96.57–96.76% top-1 agreement, p99 0.45–0.47 (noise).",
          "- Logits: 96.47–96.67% top-1 agreement, p99 0.47–0.50 against the three original captures; median difference\n"
          "  0.0004, as everywhere else. Slightly outside the original-vs-original range on both measures, so flagged\n"
          "  like the combine-norm result (§13.13)."))

# TMA baselines
E.append(("Correct, but: TMA K+V 26.03 ms (0.54×), TMA K + register V\n12.46 ms (1.12×), vs pipelined 8.69 ms.",
          "Correct, but slower than the pipelined kernel (8.69 ms): TMA K+V 26.03 ms (0.33× of pipelined; 0.54× of the\n"
          "original 14.04 ms), TMA K + register V 12.46 ms (0.70× of pipelined; 1.12× of the original)."))
E.append(("per-row TMA bulk-copy gather (0.54–1.12×)", "per-row TMA bulk-copy gather (0.33–0.70× of the pipelined kernel)"))

# prefetch distance
E.append(("issued one or two tiles ahead, consumed after `P·V`.",
          "issued two and three tiles ahead (`block_table` for i+2, `selected_positions` for i+3), consumed\n  after `P·V`."))

# L2 / KV arithmetic
E.append(("At ≤ ~24K context the whole FP8 KV cache of the 12 attention layers (~16 MB at 16K for one layer pair) stays in L2: 98% hit rate.",
          "One attention layer's FP8 KV cache (2 KV heads × 256 dims × K and V = 1 KB per token, ~16 MB at 16K) fits in L2 while that layer runs: 98% hit rate in the harness."))

# MXFP8 sign / estimate
E.append(("MXFP8 weights tested (§13.4): +2–3% decode, −7% prefill; not adopted",
          "MXFP8 weights tested (§13.4): decode +2–3% per step *[estimate]*, prefill 7% slower; not adopted"))
E.append(("MXFP8 HC weights (decode +2–3%, prefill 7% slower)", "MXFP8 HC weights (decode +2–3% per step *[estimate]*, prefill 7% slower)"))

# waves
E.append(("at 2 per SM = 85 waves;", "at 2 per SM = 86 waves (85.3);"))

# stale parenthetical / undefined attribution
E.append((" (Earlier sections attributed the sm80 WMMA kernel to the router; it is these projections.)", ""))
E.append(("HyperConnection chain (attributed later, §13):", "HyperConnection chain (its kernels summed across the categories above):"))

# §15
E.append(("| A/B outputs for §13.6–13.7 |", "| A/B outputs for §13.6, §13.7 and §13.9 (earlier A/Bs are recorded only in this report) |"))
E.append(("`VLLM_QSA_FUSED_QK_NORM=1` (via `extra.env`).",
          "`VLLM_QSA_FUSED_QK_NORM=1` (via `extra.env`). The §13.7 v2 changes ride on the last two switches."))

# glossary
E.append(("## 4. Triage: approach and decision flow",
          "### 3.1 Terms used\n\n"
          "| Term | Meaning here |\n|---|---|\n"
          "| b12x | SM12x kernel library written in CuTe DSL; the vLLM fork uses it for MoE, dense GEMM, attention, GDN and HC |\n"
          "| QSA | the model's sparse full attention: a learned indexer picks ≤ 2,051 positions per query (§3) |\n"
          "| GDN | Gated DeltaNet, the linear-attention layers (36 of 48) |\n"
          "| HC | HyperConnection: 4 residual streams per token, mixed twice per layer (§13.4) |\n"
          "| MTP | multi-token prediction: a draft layer proposing 4 tokens per step for speculative decoding |\n"
          "| TTFT | time to first token; c=1 / c=4 = 1 or 4 concurrent requests |\n"
          "| Chunk | prefill runs in 4,096-token batches, so a 16K prompt passes every layer 4 times |\n"
          "| CTA | cooperative thread array, i.e. a thread block |\n"
          "| LSE | log-sum-exp of a query's attention scores, kept so partial results can be merged |\n"
          "| NVFP4, A4 | 4-bit float weights with FP8 block scales; A4 = activations also quantised to FP4 |\n"
          "| `nvjet` | cuBLAS's closed-source GEMM kernels for this GPU |\n"
          "| GB20B | the chip name Nsight tools use for GB10's GPU |\n"
          "| `shared_N`, `disjoint` | synthetic routing patterns in b12x's `workloads.py` that its tuner measures, from heavily shared experts to evenly spread |\n"
          "| Stall reasons | Nsight Compute's per-warp wait causes: *sleeping* (parked in a wait/nanosleep loop), *barrier* (block barrier), *long scoreboard* (global/L2 load), `mio_throttle` (shared-memory pipe full), `lg_throttle` (global/atomic queue full) |\n\n"
          "## 4. Triage: approach and decision flow"))

# equivalence summary section (before §14)
E.append(("Nothing new for this workload on GB10, so no benchmark was run.\n",
          "Nothing new for this workload on GB10, so no benchmark was run.\n\n"
          "### 13.13 Equivalence summary\n\n"
          "Every capture compared with `lp_compare.py` against the three captures of the original build (A, A2 after a\n"
          "restart, A3 in the same process as A2): teacher-forced top-1 agreement over 6,975 positions and the p99 of\n"
          "the top-1 log-prob difference where top-1 agrees. Median difference is 0.0004–0.0005 in every row.\n\n"
          "| Arm | Top-1 agreement | p99 | Reading |\n|---|---|---|---|\n"
          "| Original vs original (3 pairs) | 96.72–96.76% | 0.44–0.47 | the noise range |\n"
          "| Patched attention (§9) | 96.56–96.70% | 0.46 | slightly below on top-1 |\n"
          "| + fused HC gate (§13.4) | 96.67–96.85% | 0.42–0.48 | inside |\n"
          "| + Triton combine-norm (§13.4) | 96.46–96.62% | 0.43–0.47 | top-1 up to 0.3 points below |\n"
          "| MXFP8 HC weights, not adopted (§13.4) | 96.54–96.77% | 0.44–0.45 | inside |\n"
          "| + GDN identity fast path (§13.6) | 96.79–96.97% | 0.43–0.46 | inside |\n"
          "| + fused q/k norm (§13.7) | 96.77–96.89% | 0.43–0.45 | inside |\n"
          "| + v2 cleanup (§13.7) | 96.47–96.67% | 0.47–0.50 | slightly outside on both |\n\n"
          "No arm drifts further from the original as changes accumulate: the fast path and fused q/k norm captures\n"
          "agree with the original better than the original's own pairs do. With only three control pairs, the noise\n"
          "range itself is uncertain by a few tenths of a point. The two flagged arms differ by about as much as the\n"
          "spread between unflagged arms, which reads as noise, not a systematic change. A larger control set (more\n"
          "restarts of one build) would settle it *[not done]*.\n"))

bad = []
for old, new in E:
    n = s.count(old)
    if n != 1:
        bad.append((n, old[:100])); continue
    s = s.replace(old, new)
if bad:
    for n, o in bad: print(f"MATCH {n}: {o!r}")
    sys.exit(1)
open(P, "w").write(s)
print(f"applied {len(E)} edits")

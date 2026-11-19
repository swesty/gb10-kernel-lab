# Upstream triage

These are the findings from this project, sorted by where they belong and what I plan to do with each one. I checked
every finding on 2026-11-19 against the current code of each repository it could apply to, not only against the image I
tested (standalone b12x 1.5.0 plus the vLLM fork at `ab86b707`). I also searched each project for existing issues and
PRs, and read both projects' contribution rules. Nothing has been filed yet.

## Filing plan

| # | Finding | Action | Where | Draft | Status |
|---|---|---|---|---|---|
| B1 | MTP feedback prefill tile (`_TILE_N = 16`, M-major grid) | **PR** | flashinfer-ai/flashinfer | [pr-b1-mtp-prefill-tile.md](pr-b1-mtp-prefill-tile.md), [patch](pr-b1-mtp-prefill-tile.diff); issue text kept in [b12x-01](b12x-01-mtp-prefill-tile.md) | draft ready; see the PR draft for validation |
| B2 | BF16 swizzle on element offsets (selected-position attention) | **PR** (layout atoms only) | flashinfer-ai/flashinfer | [pr-b2-selected-forward-swizzle.md](pr-b2-selected-forward-swizzle.md), [patch](pr-b2-selected-forward-swizzle.diff); issue text kept in [b12x-02](b12x-02-bf16-swizzle.md) | draft ready; see the PR draft for validation |
| B2b | Store order, software pipelining and split QK in the same kernel | later PR, after B2 | flashinfer-ai/flashinfer | none yet | bigger change; wants a design discussion |
| B2c | Same swizzle construction in `gemm/bf16_gemv/_kernel.py:389–390` | measure, then fold into an issue or PR | flashinfer-ai/flashinfer | none yet | not measured |
| B3 | Dense GEMM tile order; no tile choice in quantized mode | **issue** | flashinfer-ai/flashinfer | [b12x-03](b12x-03-dense-gemm-raster.md) | ready; how to expose it in the tuner is the maintainers' call |
| B4 | PLE `DiskTable` reads synchronously; queue depth 64 | **comment** on existing work, no new issue | [local-inference-lab/flashinfer#44](https://github.com/local-inference-lab/flashinfer/pull/44) (host row cache plus `DiskTable.prefetch`) | numbers in [b12x-04](b12x-04-ple-disk-prefetch.md) | overlaps; FlashInfer #6358 was closed in favour of #44 |
| B5 | `NCCL_IB_GID_INDEX=-1` taken literally | small **PR** (one-line fix plus a CPU test) | flashinfer-ai/flashinfer | item 1 of [b12x-05](b12x-05-small-bugs.md) | not started |
| B6, B7 | `ready_queue` compile error; deterministic M = 1 plan | re-run on FlashInfer main first; file only what reproduces | flashinfer-ai/flashinfer | items 2–3 of [b12x-05](b12x-05-small-bugs.md) | not re-run |
| B8 | Preparation cache keyed by the model snapshot path | **issue** | flashinfer-ai/flashinfer | item 4 of [b12x-05](b12x-05-small-bugs.md) | ready |
| V1–V5 | vLLM fork prefill wins | **comment**; put the PLE numbers on fork PR [#1044](https://github.com/local-inference-lab/vllm/pull/1044) rather than duplicating it | local-inference-lab/vllm | [vllm-662-comment.md](vllm-662-comment.md) | optional: the fork looks dormant (one maintainer merges, last push 2026-10-27, outside PRs open since late October, #662 has no comments) |
| M1 | Mainline PLE disk offload reads synchronously | **comment** on RFC [#66467](https://github.com/vllm-project/vllm/issues/66467) with GB10 data, and test its PRs #66464 and #66941 on a DGX Spark | vllm-project/vllm | numbers in [vllm-mainline-ple-disk.md](vllm-mainline-ple-disk.md) | do not file a new issue: the RFC covers it (its data is from a GB300) |

None of B1, B2, B3, B5, B6, B7 or B8 has an existing issue or PR in FlashInfer (searched 2026-11-19).

## What the projects require

**FlashInfer** (`CONTRIBUTING.md`, `AGENTS.md`, `flashinfer/experimental/README.md`, the PR template):

- Use the default PR template unchanged. Titles in this area follow the `fix(experimental/b12x): …` or `perf(b12x): …`
  pattern.
- b12x lives under `flashinfer/experimental/`, so PRs tick the **Experimental Track** box and link the b12x tracking
  issue, [#6909](https://github.com/flashinfer-ai/flashinfer/issues/6909) (opened by Luke Alonso). Small improvements to an
  existing experimental backend do not need a new tracking issue, and get a narrower review (correctness, containment,
  licensing).
- Tests go under `tests/experimental/`, validated on the intended hardware. Performance claims need reproducible
  before and after numbers with the GPU, problem sizes and cache conditions, ideally from
  `flashinfer.testing.bench_gpu_time` with CUPTI.
- Kernel specialization changes must be reflected in the compile-cache key.
- AI-assisted PRs are welcome, but the author has to understand the change and walk a reviewer through it on request.
  Mention the AI assistance.
- No DCO or CLA. CI does not start on its own: someone with label rights or a `ci-users` member runs
  `@flashinfer-bot run tests/<paths>`. NVIDIA's internal CI has RTX PRO 6000 and (manual) Spark runners.
- Reality check: merged b12x fixes took from the same day to about three weeks; 73 b12x PRs were open on 2026-11-19.

**vLLM** (`docs/contributing/README.md`, `AGENTS.md`, the PR template), for any later vLLM PR:

- Every commit needs a DCO `Signed-off-by:` line (`git commit -s`), which puts a name and email in public history.
- Pure agent PRs are banned ("Breaching these guidelines can result in automatic banning"). The submitter reviews every
  changed line, runs the tests, and can defend the change. The PR description states that AI was used, commits carry a
  `Co-authored-by:` trailer, and agent-prepared PRs run vLLM's `/pr-checklist` skill.
- The template asks for Overview, Claims, Validation (the exact pytest summary line and the commit SHA) and Details.
  Titles start with tags such as `[Bugfix][Kernel]`. User-visible settings, including environment variables, are
  documented in the same PR. More than 500 lines outside kernels and tests needs an RFC first. At most 6 open PRs per
  contributor.

## Where the code lives

| Repository | What it holds | Status | Relevant here |
|---|---|---|---|
| **[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)** `flashinfer/experimental/b12x/` | The b12x kernels, now maintained inside FlashInfer. Mainline vLLM's docs say b12x "ships with FlashInfer (`flashinfer-python>=0.7.2`), so no separate install is needed". | active; `main` @ `d3f59f0`. FlashInfer 0.7.3 ships the same `_cute_prefill.py` and `_selected_forward.py` as that commit | **all b12x findings** (verified present) |
| [local-inference-lab/flashinfer](https://github.com/local-inference-lab/flashinfer) | A FlashInfer fork carrying extra b12x work, including the PLE/Engram host row cache (#44) | active (pushed 2026-11-19) | B4 |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (`lukealonso/b12x` redirects here) | The original standalone b12x, still what the vLLM fork and eugr's `spark-vllm-b12x` image install (1.5.0) | **archived** (read-only) since 2026-10-06; `master` @ `2cc7f66` | all b12x findings present, but nothing can be filed here |
| [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) `dev/karmic-kraken` | The vLLM fork that eugr's b12x recipes build | `8244d1c` (2026-10-15); last push 2026-10-27 | fork-side model and integration findings |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) `main` | Mainline vLLM, with its own Qwen4-Exp implementation (`vllm/models/qwen4_exp/`) consuming b12x through FlashInfer | `4ae5147` (2026-11-19) | PLE disk offload only |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Recipes | n/a | nothing to file |

## Will the fixes regress SM120? (GB10 = sm_121 vs RTX PRO 6000 = sm_120), measured 2026-11-19

The two parts share the same SM (same ISA, `mma.sync` only, 100 KB of shared memory per SM, the same bank layout) but
have very different memory systems: GB10 has **48 SMs, 24 MB L2, ~273 GB/s** (238 GB/s achievable); the RTX PRO 6000
has **188 SMs, 128 MB L2, ~1.8 TB/s**. Measured `mma.sync` peaks on the RTX PRO 6000 are BF16 554, FP8 1,107 and NVFP4
2,218 TFLOPS, 4.45× GB10.

**How b12x picks configurations today.** b12x autotunes at preparation time (`b12x.preparation`: `Knob`,
`TuningContract`). It measures candidate configurations on the actual GPU and caches the winners per `DeviceIdentity`
(vendor, compute capability, SM count, product name), so it already tells GB10 apart from an RTX PRO 6000; L2 size is
not part of the identity. The dense GEMM also has an explicit Spark policy, `_use_low_sm_dense_tactics(sm_count)` (64
SMs or fewer). **The problem is not hardware detection: the settings I changed are hard-coded constants the tuner never
explores.**

**What I ran.** I re-measured every b12x finding on an RTX PRO 6000 (jarvis, sm_120, driver 580.173.02) with the same
b12x 1.5.0 source and the same harnesses, in `vllm-openai:v0.30.0` with cutlass-dsl 4.7.1 and torch 2.13 cu130 (spark-5
has cutlass-dsl 4.7.0). The production model was stopped for the timed runs (17:36–17:44 UTC), and the L2 flushes were
enlarged to 512 MB for the 128 MB L2. Raw data is in [`data/sm120/`](../data/sm120/) and the scripts in
[`scripts/sm120/`](../scripts/sm120/). There was one run per configuration; placement-to-placement spread on sm_120 was
1% or less, so treat differences under about 1.5% as noise.

| # | Finding | sm_120 result | Verdict | Upstream form |
|---|---|---|---|---|
| B1 | MTP prefill tile 16 + M-major | Stock 1,764 µs (1,919 after an L2 flush); my GB10 setting (tile 128, 2 stages, N-major) **684 µs (688)**, 2.6–2.8× faster, bit-identical, within 1.2% of the best sm_120 variant (128/2/M-major, 676). cuBLAS: 526 µs. N-major **alone** at tile 16 is 12% slower back-to-back (1,971 µs) | **Helps sm_120 too.** The tile is what matters; the grid order depends on the hardware and cache state | PR: default tile 128 with 2 stages and N-fastest order |
| B2 | BF16 swizzle on element offsets | Bank-conflict counts **identical to GB10 to three significant figures**. Stock: loads 46.8%, stores 51.7% of wavefronts are conflicts. `S<3,3,3>` alone: loads 0.0%, stores unchanged (51%), 2.6–5.5% faster. With the pipelined gather's store order: stores 4.7%. Full patch: 1.83× at 16K | **GPU-agnostic, as predicted** | PR: the layout atoms; the store order and pipelining later |
| B3 | Dense GEMM M-major walk; no tile choice in quantized mode | sm_120 picks different tiles (128×128×64 for the big projections, 128×64 or 64×128 elsewhere; GB10's low-SM lowering picks 64×128×128 for all). My exact recipe on sm_120: attention qkv −5.2%, gate/up −3.2%, down −5.7%, out_proj unchanged (the 20-N-tile rule does not fire on its tile), **in_proj_qkvz +2.7% slower** (large-M unroll). N-major on sm_120's own tiles: −6.5% to +1.1%. Placement spread 1.25–2.05 ms on GB10, 1% or less on sm_120 | **Mixed: hard-coding would regress one shape on sm_120** | Issue: tuner options (order and tile for quantized-mode plans) or the low-SM policy; not a new default |
| B4 | PLE disk reads synchronous; queue depth 64 | not GPU-dependent (host and NVMe); not re-run | n/a | comment on local-inference-lab/flashinfer#44 |
| B5–B8 | GID −1, ready_queue, deterministic, cache identity | not GPU-dependent; not re-run | n/a | as in the filing plan |
| V1–V5, M1 | vLLM side | not GPU-dependent kernels; not re-run | n/a | as in the filing plan |

**Bottom line.** The swizzle fix (B2) and the MTP tile (B1) improve sm_120 as much as or more than GB10, so they are safe
to send as fixes. The dense GEMM changes (B3) really are hardware-dependent, so they go in as "let the tuner choose; here
are the GB10 and sm_120 numbers", not as a default change. I measured only one sm_120 part (RTX PRO 6000); a 5090-class
part (170 SMs, 96 MB L2) was not tested.

## b12x kernel findings (FlashInfer)

local-inference-lab/b12x is **archived** (read-only since 2026-10-06; `gh repo view` reports `isArchived: true`, and its
last commit is the migration README), so nothing can be filed there. The "LIL b12x" column records that the frozen 1.5.0
code that Spark images still install has the same bugs. The two trees hold the same code; FlashInfer's copies differ only
in formatting and imports, plus unrelated additions.

| # | Finding | FlashInfer `main` | LIL b12x `master` | Evidence | Draft | Priority |
|---|---|---|---|---|---|---|
| B1 | **MTP feedback prefill GEMM**: `_TILE_N = 16` and an M-major grid re-read A once per N tile (13.4 GB per chunk) | present (`sequence/mtp_feedback/_cute_prefill.py:29`, `:280`) | present | 57.7 to 2.6 ms (22×), bit-identical; −3.7% prefill | [PR draft](pr-b1-mtp-prefill-tile.md) | **high** |
| B2 | **BF16 swizzle on element offsets**: `allocate_tensor(BFloat16, composed S<3,4,3> layout)` gives 2-way bank conflicts | present, both sites: `attention/paged/_selected_forward.py` (`_shared_layouts`), `gemm/bf16_gemv/_kernel.py:389–390` | present (same two sites) | attention (raw bank-conflict counters): loads 46.8% to 0.0%, stores 51.6% to ~5%, same on sm_120; bf16_gemv unmeasured | [PR draft](pr-b2-selected-forward-swizzle.md) | **high** |
| B3 | **Dense GEMM persistent scheduler always walks along M** (A re-streamed when A is larger than W and than L2); quantized-mode `BlockscaledQuery` has no tile choice | present (`_lib/dense_gemm.py:5054`; `_raster_along_m` appears only in a TVM-FFI compatibility shim) | present | out_proj −23 to −30%, qkvz −4 to −10% | [b12x-03](b12x-03-dense-gemm-raster.md) | medium |
| B4 | **PLE `DiskTable` reads synchronously in the batch**; `queue_depth` default 64 | present (`sequence/ple_embedding/_disk.py:19`, `sequence/_shared/disk_table.py:331`); the Engram disk table already has no-op `prefetch` / `abort_prefetch` hooks | present | QD 256: −4.1% TTFT; next-chunk prefetch: −6.2% | [b12x-04](b12x-04-ple-disk-prefetch.md) | medium |
| B5 | `NCCL_IB_GID_INDEX=-1` ("auto") taken literally as GID −1 | present (`comm/roce/roce_oneshot.py:88`) | present | by inspection | [b12x-05](b12x-05-small-bugs.md) | low to medium |
| B6 | `B12X_DYNAMIC_WORK_SOURCE=ready_queue` compile error | not re-run | not re-run | 1.5.0 | b12x-05 | low |
| B7 | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` cannot prepare the M = 1 plan | FlashInfer has recent deterministic-output fixes (`ecc4b8a9`, 2026-11-05), so it **may be fixed**; check before filing | not re-run | 1.5.0 | b12x-05 | low |
| B8 | Preparation-cache identity keyed by the model snapshot path, so a new Hugging Face revision triggers a cold autotune | not checked in FlashInfer's tree | present | REPORT §13.20 | b12x-05 | low |
| B9 | In-GEMM MXFP8 quantization only for M ≤ 8 | present | present | context for B3 | in b12x-03 | low |

## vLLM fork (`local-inference-lab/vllm`)

All of these are still present on `dev/karmic-kraken`, and all are prefill wins on Qwen3.8-Flash-Next
([draft comment](vllm-662-comment.md)).

| # | Finding | Gain | Related work | In mainline vLLM? |
|---|---|---|---|---|
| V1 | Pass `queue_depth` to b12x `DiskTable` (default 64) and prefetch the next prefill chunk's PLE rows | −4.1% and −6.2% TTFT | fork PR [#1044](https://github.com/local-inference-lab/vllm/pull/1044) (advisory row-cache prefetch, built on local-inference-lab/flashinfer#44) | different code (see M1) |
| V2 | `B12xQSAAttention` doesn't use the fused q/k norm + mRoPE + gate kernel | −5% | none found | **already done** (`use_fused_qk_norm_rope_gate`, `qsa_prepare`) |
| V3 | The GDN b12x mixed path always gathers and scatters, even for the identity worklist | −4% | #661 may cover part of it | n/a (mainline has no b12x mixed GDN path) |
| V4 | HyperConnection: fused up-projection + gated mean; Triton combine-norm | −5% | related to #659 | **already done in its own form** (`hc_up_gate_mix` CUDA kernel, `hc_combine_norm`) |
| V5 | MXFP8 activation quantization fused into its producers (needs a b12x hook) | −1.5% prefill, −2.0% TTFT | none found | not checked |

## Mainline vLLM (`vllm-project/vllm`)

| # | Finding | Status | Related work |
|---|---|---|---|
| M1 | **PLE disk offload reads rows synchronously in the step, one serial `os.preadv` per unique row in Python** (after `POSIX_FADV_WILLNEED` readahead), and `start_prefetch` is a no-op for the disk backend ("PLE disk offload does not support prefetch"). The b12x queue-depth bug does not exist here (there is no `io_uring`), but the underlying problem, the GPU idling while the host reads, does | present (`vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, `Qwen4ExpPLEDiskOffloadEmbedding._read_rows`, `start_prefetch`); **not measured on mainline** | RFC [#66467](https://github.com/vllm-project/vllm/issues/66467) with PR [#66464](https://github.com/vllm-project/vllm/pull/66464) (batched `O_DIRECT` io_uring reader) and PR [#66941](https://github.com/vllm-project/vllm/pull/66941) (reads the next prefill chunk's rows during the current step). Same idea as mine, measured on a GB300. My useful contribution is DGX Spark data |

## Data point for an existing issue

[local-inference-lab/b12x#444](https://github.com/local-inference-lab/b12x/issues/444) (crash with more than 3 parallel
Flash Next requests, dual Spark): on one Spark (TP1, b12x 1.5.0 image) my soak test ran 8-way concurrency with mixed
1–16K prompts without errors (REPORT §13.20). The repository is archived, so this can no longer be posted there; I will
mention it in a FlashInfer issue if it becomes relevant.

## Not upstream issues

- An online `vllm serve` or `vllm bench serve` moving the Hugging Face `refs/main` is normal Hub behaviour; use
  `HF_HUB_OFFLINE=1`.
- KV cache size varying by launch path is expected from `gpu_memory_utilization`; my recipe pins
  `--kv-cache-memory-bytes`.

# Upstream triage

Findings from this project, sorted by where they belong. Each was checked on 2026-11-19 against the current code of
every repository it could apply to, not only against the image we tested (standalone b12x 1.5.0 + vLLM fork `ab86b707`).
Nothing below has been filed; drafts are in this folder.

## Where the code lives

| Repository | What it holds | Status | Relevant here |
|---|---|---|---|
| **[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)** `flashinfer/experimental/b12x/` | The b12x kernels, now maintained inside FlashInfer. Mainline vLLM's docs say b12x "ships with FlashInfer (`flashinfer-python>=0.7.2`), so no separate install is needed". | active: several contributors, last b12x commits 2026-11-05/06; `main` @ `d3f59f0` | **all b12x findings** (verified present) |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (`lukealonso/b12x` redirects here) | The original standalone b12x; still what the vLLM fork and eugr's `spark-vllm-b12x` image install (1.5.0) | last push 2026-10-06; `master` @ `2cc7f66` | all b12x findings (verified present) |
| [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) `dev/karmic-kraken` | The vLLM fork eugr's b12x recipes build (Qwen3.8-Flash-Next model code using standalone b12x) | `8244d1c` (2026-10-15) | fork-side model/integration findings |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) `main` | Mainline vLLM. Has its **own** Qwen4-Exp implementation (`vllm/models/qwen4_exp/`), consuming b12x through FlashInfer | `4ae5147` (2026-11-19) | PLE disk offload only (see below) |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Recipes | — | nothing to file |

## Will the fixes regress SM120? (GB10 = sm_121 vs RTX PRO 6000 = sm_120)

Same SM architecture (same ISA, `mma.sync` only, 100 KB shared memory per SM, same bank layout), very different memory
system: GB10 has **48 SMs, 24 MB L2, ~273 GB/s** (238 GB/s achievable); the RTX PRO 6000 has **188 SMs, 128 MB L2,
~1.8 TB/s**. Some findings are pure bugs that behave the same on both; others are tuning choices whose best value depends on
SM count and L2 size, and those must not be "fixed" with GB10 constants.

**How b12x picks configurations today.** b12x autotunes at preparation time (`b12x.preparation`: `Knob`, `TuningContract`):
it measures candidate configs on the actual GPU and caches the winners per `DeviceIdentity` (vendor, compute capability,
SM count, product name — so it already distinguishes GB10 from an RTX PRO 6000; L2 size is not part of the identity). The dense
GEMM also has an explicit Spark policy, `_use_low_sm_dense_tactics(sm_count)` (≤ 64 SMs). **The problem is not hardware
detection: the settings we changed are hard-coded constants the tuner never explores.** So the right upstream fix for the
hardware-dependent items is to make them tunable (or gate them on the existing low-SM policy), letting each GPU measure its own
best choice.

| # | Finding | Depends on GPU memory system? | SM120 risk of our fix as-is | Upstream form that cannot regress SM120 | Validated on |
|---|---|---|---|---|---|
| B1 | MTP prefill tile 16 + M-major | **Yes.** The 84 MB A operand streams from DRAM on GB10 (24 MB L2) but fits in the RTX PRO 6000's 128 MB L2, so the penalty there is mostly L2 traffic and low arithmetic intensity, not DRAM | Low (all variants bit-identical; a 128-wide tile is the normal GEMM choice), but unmeasured | Add `tile_n` ∈ {16, 64, 128} and `raster` ∈ {M, N} as knobs in `MtpFeedbackConfig` (the contract already tunes the norm params); the tuner picks per device | sm_121 only |
| B2 | BF16 swizzle on element offsets | **No.** Bank conflicts depend on the SM's shared-memory banks, identical on sm_120 and sm_121 | Very low: the fix only removes conflicts (wavefront counts are architecture-independent within SM12x) | Plain bug fix | sm_121 only |
| B3 | Dense GEMM M-major walk; no tile knob in quantized mode | **Yes.** Whether A re-streams from DRAM depends on L2 size; best tile depends on SM count | **Real if hard-coded**: on 128 MB L2, out_proj's 25 MB A already fits, so forcing N-major could be neutral or slower | Add `raster_along_m` and the tile options to the tuner's candidate set for quantized-mode plans, or apply them only under the existing low-SM (Spark) policy | sm_121 only |
| B4 | PLE disk reads synchronous; queue depth 64 | **No** (host/NVMe-bound; GPU-agnostic) | None: prefetched rows are used only on an exact ID match; queue depth affects only disk I/O | Prefetch API + configurable queue depth (best value is NVMe-dependent) | sm_121 only |
| B5–B8 | GID −1, ready_queue, deterministic, cache identity | No | None | Plain fixes | — |
| V1–V5 | vLLM fork: PLE queue depth/prefetch, fused q/k norm, GDN identity path, HC fusions, quantisation fusion | No: they remove copies and passes, not retune kernels | Very low (logprob-equivalent on sm_121; expected to help sm_120 too) | As drafted | sm_121 only |
| M1 | mainline vLLM PLE disk reads | No | None | As drafted | — |

**Before filing**, either validate the SM120 side (B1 and B3 above all, plus a quick B2 check) on an RTX PRO 6000, or file
B1/B3 as "tunable choice; here is GB10 data" rather than "change the default", and state plainly that only sm_121 was measured.

## b12x kernel findings → file at **FlashInfer** and at **local-inference-lab/b12x**

Same code in both trees (FlashInfer's copies differ only in formatting/imports, plus unrelated additions).

| # | Finding | FlashInfer `main` | LIL b12x `master` | Evidence | Draft | Priority |
|---|---|---|---|---|---|---|
| B1 | **MTP feedback prefill GEMM** `_TILE_N = 16` + M-major grid → A re-read once per N tile (13.4 GB per chunk) | present (`sequence/mtp_feedback/_cute_prefill.py:29`, `:280`) | present | 57.7 → 2.6 ms (22×), bit-identical; −3.7% prefill | [b12x-01](b12x-01-mtp-prefill-tile.md) | **high** |
| B2 | **BF16 swizzle on element offsets**: `allocate_tensor(BFloat16, composed S<3,4,3> layout)` → 2-way bank conflicts | present, both sites: `attention/paged/_selected_forward.py` (`_shared_layouts`, `allocate_tensor` at :616–631), `gemm/bf16_gemv/_kernel.py:389–390` | present (same two sites) | attention: 43–54% excess wavefronts → 0%; bf16_gemv unmeasured | [b12x-02](b12x-02-bf16-swizzle.md) | **high** |
| B3 | **Dense GEMM persistent scheduler always M-major** (A re-streamed when A > W and > L2); quantized-mode `BlockscaledQuery` has no tile knob | present (`_lib/dense_gemm.py:5054`; `_raster_along_m` appears only in a TVM-FFI compat shim) | present | out_proj −23–30%, qkvz −4–10% | [b12x-03](b12x-03-dense-gemm-raster.md) | medium |
| B4 | **PLE `DiskTable` reads synchronously in the batch**; `queue_depth` default 64 | present (`sequence/ple_embedding/_disk.py:19`, `sequence/_shared/disk_table.py:331`) | present | QD 256: −4.1% TTFT; next-chunk prefetch: −6.2% | [b12x-04](b12x-04-ple-disk-prefetch.md) | medium |
| B5 | `NCCL_IB_GID_INDEX=-1` ("auto") taken literally as GID −1 | present (`comm/roce/roce_oneshot.py:88`) | present | by inspection | [b12x-05](b12x-05-small-bugs.md) | low–medium |
| B6 | `B12X_DYNAMIC_WORK_SOURCE=ready_queue` compile error | not re-run | not re-run | 1.5.0 | b12x-05 | low |
| B7 | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` cannot prepare the M = 1 plan | FlashInfer has recent deterministic-output fixes (`ecc4b8a9`, 2026-11-05); **may be fixed** — check before filing | not re-run | 1.5.0 | b12x-05 | low |
| B8 | Preparation-cache identity keyed by model snapshot path → cold autotune on a new HF revision | not checked in FlashInfer's tree | present | §13.20 | b12x-05 | low |
| B9 | In-GEMM MXFP8 quantisation only for M ≤ 8 | present | present | context for B3 | in b12x-03 | low |

## vLLM fork (`local-inference-lab/vllm`) → one comment on [#662](https://github.com/local-inference-lab/vllm/issues/662)

All still present on `dev/karmic-kraken`; all are prefill wins on Qwen3.8-Flash-Next ([draft](vllm-662-comment.md)).

| # | Finding | Gain | In mainline vLLM? |
|---|---|---|---|
| V1 | Pass `queue_depth` to b12x `DiskTable` (default 64) + prefetch next prefill chunk's PLE rows | −4.1% + −6.2% TTFT | **different code** (see M1) |
| V2 | `B12xQSAAttention` doesn't use the fused q/k norm + mRoPE + gate kernel | −5% | **already done** (`use_fused_qk_norm_rope_gate`, `qsa_prepare`) |
| V3 | GDN b12x mixed path always gathers/scatters even for the identity worklist | −4% | **n/a** (mainline has no b12x mixed GDN path) |
| V4 | HyperConnection: fused up-projection + gated mean; Triton combine-norm | −5% | **already done in its own form** (`hc_up_gate_mix` CUDA kernel, `hc_combine_norm`) |
| V5 | MXFP8 activation quantisation fused into its producers (needs a b12x hook) | −2% | not checked |

## Mainline vLLM (`vllm-project/vllm`) → one issue

| # | Finding | Status | Draft |
|---|---|---|---|
| M1 | **PLE disk offload reads rows synchronously in the step, one serial `os.preadv` per unique row in Python** (after `POSIX_FADV_WILLNEED` readahead), and `start_prefetch` is a no-op for the disk backend ("PLE disk offload does not support prefetch"). The b12x queue-depth bug itself does not exist here (no `io_uring`), but the underlying problem — GPU idle while the host reads — does, and likely worse. | present (`vllm/models/qwen4_exp/nvidia/ngram_embedding.py`, `Qwen4ExpPLEDiskOffloadEmbedding._read_rows`, `start_prefetch`) — **not measured on mainline** | [vllm-mainline-ple-disk.md](vllm-mainline-ple-disk.md) |

## Data point for an existing issue

[local-inference-lab/b12x#444](https://github.com/local-inference-lab/b12x/issues/444) (crash with > 3 parallel Flash Next
requests, dual Spark): on one Spark (TP1, b12x 1.5.0 image) our soak ran 8-way concurrency with mixed 1–16K prompts
without errors (REPORT §13.20).

## Not upstream issues

- An online `vllm serve` / `vllm bench serve` moving the Hugging Face `refs/main` — normal Hub behaviour; use `HF_HUB_OFFLINE=1`.
- KV cache size varying by launch path — expected from `gpu_memory_utilization`; our recipe pins `--kv-cache-memory-bytes`.

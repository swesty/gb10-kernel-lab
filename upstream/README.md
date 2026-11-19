# Upstream triage

Findings from this project, sorted by where they belong. Verified on 2026-11-19 against the current upstream
code, not just the image we tested (b12x 1.5.0 + vLLM `ab86b707`, Oct 4):

- **b12x** — <https://github.com/local-inference-lab/b12x> (`lukealonso/b12x` redirects here), `master` @ `2cc7f66` (Oct 6).
  The kernels: attention, GEMM, MoE, GDN, PLE disk table, comm.
- **vLLM fork** — <https://github.com/local-inference-lab/vllm>, `dev/karmic-kraken` @ `8244d1c` (Oct 15; still the ref
  eugr's spark-vllm-docker builds). Model code (`vllm/models/qwen4_exp/nvidia/`), GDN layer, b12x integration.
- **spark-vllm-docker** — <https://github.com/eugr/spark-vllm-docker>. Recipes only; nothing here is a bug of theirs.

Nothing below has been filed. Drafts are in this folder.

## b12x

| # | Finding | Kind | Still on master? | Evidence | Suggested action | Priority |
|---|---|---|---|---|---|---|
| B1 | **MTP feedback prefill GEMM**: `_TILE_N = 16` + M-major grid re-reads the 84 MB A operand once per N tile (13.4 GB per 4,096-token chunk) | perf bug | **yes** (`sequence/mtp_feedback/_cute_prefill.py:29`) | 57.7 → 2.6 ms (22×), bit-identical; −3.7% prefill (REPORT §13.15) | issue + PR ([draft](b12x-01-mtp-prefill-tile.md)) | **high** |
| B2 | **BF16 smem swizzle applied to element offsets**: `allocate_tensor(BFloat16, composed_layout)` with the 128-byte atom `S<3,4,3>` → every 128-bit store and `ldmatrix` 2-way bank-conflicted | perf bug | **yes**, two sites: `attention/paged/_selected_forward.py` (Q/K/V/P tiles) and `gemm/bf16_gemv/_kernel.py:389–390` | attention: 46–54% excess wavefronts → 0% with `S<3,3,3>`; part of a 1.9× kernel gain (REPORT §9.7). bf16_gemv: same pattern, not measured | issue ([draft](b12x-02-bf16-swizzle.md)) | **high** |
| B3 | **Dense GEMM persistent scheduler always walks along M**: when A is larger than W and than L2, A is re-streamed from DRAM per N column; also quantized-mode `BlockscaledQuery` exposes no tile knob, so the tuner cannot reach 128×128/k64 | perf | **yes** (`_lib/dense_gemm.py` `_compute_grid`; `gemm/blockscaled/_preparation.py`) | out_proj 4096×2560×6144: −23–30% with N-major; 1.25–2.05 ms placement variance disappears; qkvz −4–10% with 128×128/k64 (§13.16) | issue ([draft](b12x-03-dense-gemm-raster.md)) | medium |
| B4 | **PLE `DiskTable` reads synchronously inside the batch** (hash → host sync → `io_uring` read → lookup); GPU idle ~110–190 ms per prefill chunk | design / feature | yes (`sequence/_shared/disk_table.py`, `ple_embedding/_disk.py`) | prefetching the next chunk's rows behind the current chunk: −6.2% TTFT; queue depth 256: −4.1% (§13.17) | issue proposing an async/prefetch API ([draft](b12x-04-ple-disk-prefetch.md)), pairs with V1 | medium |
| B5 | `B12X_ROCE_GID_INDEX` falls back to `NCCL_IB_GID_INDEX` literally, so NCCL's `-1` ("auto") becomes GID index −1 | bug | yes, by inspection (`comm/roce/roce_oneshot.py:88`) | found earlier on the fleet; worked around locally | issue ([small-bugs draft](b12x-05-small-bugs.md)) | low–medium |
| B6 | `B12X_DYNAMIC_WORK_SOURCE=ready_queue` fails to compile (`PHASE_DYNAMIC_TO_STATIC_BOOL`, "Cannot use a Runtime value … where a Python value boolean is required") | bug | probably (master's `dynamic.py` changes do not touch that path; not re-run) | §13.2 | same draft | low |
| B7 | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1`: the M = 1 decode plan cannot be prepared, so vLLM fails to start | bug | not re-run | §13.11 | same draft | low |
| B8 | Preparation cache identity includes the resolved **model snapshot path**: a new HF revision with byte-identical weights triggers a full cold autotune (and memory pressure) | feature | yes (`identity.namespace.model`) | §13.20 | same draft | low |
| B9 | In-GEMM MXFP8 activation quantisation (`dense_gemm_fused_quant_a`) only supports M ≤ 8; prefill pays a separate `_quantize` pass | feature | yes | we fused into producers via a workspace hook instead: −2% (§13.19) | mention in B3 issue | low |

## vLLM fork (`local-inference-lab/vllm`)

All are prefill wins on Qwen3.8-Flash-Next; each is behind an env switch in our patches and verified with
teacher-forced logprobs. The fork already tracks Flash Next optimisation in
**[#662](https://github.com/local-inference-lab/vllm/issues/662)** ("optimization review and merge checklist"), so the
recommendation is **one summary comment there** ([draft](vllm-662-comment.md)) plus PRs for the self-contained ones.

| # | Finding | Still on `dev/karmic-kraken`? | Gain | Overlap with existing work | Suggested action |
|---|---|---|---|---|---|
| V1 | Pass `queue_depth` to `api.DiskTable` (stock 64) and prefetch the next prefill chunk's PLE rows | **yes** (`b12x_ple.py:185`, no prefetch) | −4.1% + −6.2% TTFT | none found | PR (queue depth is a one-liner; prefetch needs B4 or a local thread) |
| V2 | `B12xQSAAttention` never uses vLLM's existing `fused_qk_rmsnorm_rope_gate`; eager `GemmaRMSNorm` chain + `output * sigmoid(gate)` | **yes** (`b12x_qsa.py:953`, `:2692`, `:2725`) | −5% prefill | none found | PR |
| V3 | GDN `_forward_core_b12x_mixed` always gathers/scatters + zero-fills, even when the worklist is the identity (plain prefill) | **yes** (unchanged since our image) | −4% prefill | #661 ("direct caller-owned GDN/PLE views") may cover part of it — check before a PR | comment / PR after checking #661 |
| V4 | HyperConnection: fuse up-projection + gated mean (Triton); faster combine-norm for prefill | **yes** (`hyperconnection.py` unchanged) | −5% prefill | #659 (projection slices without staging copies) is related but different | comment, PR if wanted |
| V5 | Fuse MXFP8 activation quantisation into the GDN gated norm and attention gate (needs a b12x workspace hook) | yes | −2% prefill | depends on B3/B9 | comment only |

Also useful as a data point on [b12x#444](https://github.com/local-inference-lab/b12x/issues/444) (crash with > 3
parallel Flash Next requests, dual Spark): on one Spark (TP1, b12x 1.5.0 image) our soak ran 8-way concurrency with
mixed 1–16K prompts without errors (§13.20).

## Not upstream issues

- `vllm bench serve` / an online `vllm serve` moving the Hugging Face `refs/main` — normal Hub behaviour; run benchmarks
  with `HF_HUB_OFFLINE=1`.
- KV cache size varying by launch path — expected from `gpu_memory_utilization`; our recipe pins
  `--kv-cache-memory-bytes`.

# gb10-kernel-lab

Profiling and kernel work to speed up **Qwen3.8-Flash-Next-NVFP4** (vLLM b12x fork + b12x 1.5.0) on an
**NVIDIA DGX Spark (GB10, sm_121)**, from live-traffic triage down to individual CUDA/CuTe DSL/Triton kernels.

| On one DGX Spark, same session | Stock | Optimised | Change |
|---|---|---|---|
| 14.5K-token real-text prompt, prefill | 6.61 s | 4.76 s | **−28%** (~2,200 → ~3,060 tokens/s) |
| 16K-token prompt, time to first token | 7.09 s | 4.89 s | **−31%** |
| Decode | | | unchanged (memory-bandwidth-bound) |

Every change is behind an environment switch. Each was checked against the engine's own run-to-run noise
with teacher-forced log-probabilities, and the full set passed a mixed-load soak (77/77 requests, 32K and
128K-token prompts, 0 errors).

## What is here

| Path | Contents |
|---|---|
| [`REPORT.md`](REPORT.md) | The full write-up: triage flow, hardware facts measured on GB10, each investigation (what, why, how, outcome), kernel deep dives, rejected ideas, corrections, tooling lessons |
| [`recipe/`](recipe/) | **Try it yourself**: a [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) recipe + mod that installs the changes (checksum-gated to the tested image), with a bench script |
| [`patches/`](patches/) | Diffs per change, plus a snapshot of every patched file as deployed next to its original |
| [`scripts/`](scripts/) | Harnesses (attention, MoE, MTP GEMM, dense GEMM), microbenchmarks (`mma_peak.cu`, `bw.py`), trace analysis (`gaps.py`, `cats.py`), A/B and logprob tools, patch writers |
| [`harness/attention/`](harness/attention/) | The selected-position attention harness and the patched kernel |
| [`data/`](data/) | Nsight Systems and Nsight Compute reports, A/B outputs, logprob captures, live MoE routes, GEMM sweeps |
| [`reference/b12x-1.5.0/`](reference/b12x-1.5.0/) | Unmodified b12x source as deployed, for reading alongside the report |
| [`upstream/`](upstream/) | Triage of the bugs and improvements for b12x and the vLLM fork, with issue drafts (checked against current upstream) |
| `work/` | Raw working files from the investigation |

Two large files are attached to the GitHub release instead of the tree: the 4-layer side-engine Nsight
Compute report (`data/ncu/side-engine/main.ncu-rep`, 150 MB) and the live QSA selection dumps
(`data/qsa-dump/sel_*.pt`, 8 × 34 MB).

## Highlights

- A **2-way shared-memory bank conflict** in b12x's BF16 tiles: a 128-byte swizzle atom applied to element
  offsets. Fixing it, plus pipelining and spreading QK and softmax over all four warps, made the attention
  kernel 1.9× faster.
- An **MTP projection kernel re-reading 13 GB per prefill chunk** because of a 16-column tile and M-major
  rasterisation: 57.7 ms → 2.6 ms, bit-identical.
- **~0.7 s of idle GPU per 16K prefill**, hidden from every kernel-time table: synchronous reads of a 28.9 GB
  n-gram embedding table from NVMe. A deeper `io_uring` queue plus next-chunk prefetch cut TTFT by 10%.
- Measured GB10 tensor-core peaks (BF16 124, FP8 249, NVFP4 498 TFLOPS with `mma.sync`) and achievable
  memory bandwidth (~238 GB/s), used to size every remaining opportunity before spending effort on it.

## Licence

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Contains modified copies of vLLM and b12x
source files, both Apache-2.0. Unofficial; not affiliated with NVIDIA, vLLM, b12x or local-inference-lab.

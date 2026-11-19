<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x) and local-inference-lab/b12x. Not filed. -->
# MTP feedback prefill projection re-reads its A operand once per 16-column tile (22× slower than needed)

**Where:** `sequence/mtp_feedback/_cute_prefill.py` (`MTPPrefillBf16GemmKernel`, `_TILE_N` at line 29, `block_idx` at 280) —
present in FlashInfer `main` @ `d3f59f0` (`flashinfer/experimental/b12x/`) and in local-inference-lab/b12x `master` @ `2cc7f66`.

## What happens

The hidden projection of `mtp_feedback` is `[tokens × 4, 2560] × [2560, 2560]ᵀ` plus a broadcast add of the token path.
For a 4,096-token prefill chunk that is 16,384 × 2,560 × 2,560 (215 GFLOP). The kernel hard-codes `_TILE_N = 16` with
`tile_m = 128` and launches `grid = (M tiles, N tiles)`, so blocks that share an A row-panel are spread across the whole
launch. The 84 MB BF16 A operand does not fit in L2 (24 MB on GB10), so it is streamed from memory once per N tile:
160 × 84 MB ≈ 13.4 GB per call. At ~220 GB/s that is the observed ~60 ms.

On a DGX Spark (GB10, sm_121) serving Qwen3.8-Flash-Next-NVFP4 this one kernel took **61 ms per prefill chunk**
(3.5 TFLOPS; measured BF16 `mma.sync` peak on this GPU is 124 TFLOPS), about 230 ms of a 16K-token prefill.

## Measurements (standalone harness, M = 16,384, output checked against a PyTorch BF16 reference)

| Tile N | Stages | Grid raster | Time | TFLOPS |
|---|---|---|---|---|
| 16 (current) | 3 | M fastest | 57,728 µs | 3.7 |
| 16 | 3 | N fastest | 8,419 µs | 25.5 |
| 64 | 3 | M fastest | 15,240 µs | 14.1 |
| 64 | 3 | N fastest | 3,223 µs | 66.6 |
| 128 | 3 | N fastest | 2,646 µs | 81.1 |
| **128** | **2** | **N fastest** | **2,619 µs (22×)** | **82.0** |
| cuBLAS BF16 matmul, same shape, no epilogue | | | 2,150–2,340 µs | 92–100 |

Every variant's output is **bit-identical** to the BF16 reference. Live, with tile N 128 / 2 stages / N-fastest:
real-text 14.5K prefill 5.47 → 5.27 s (−3.7%), 16K TTFT 5,842 → 5,639 ms (−3.5%), decode unchanged.

## Hardware dependence

Measured on sm_121 only. How much this costs depends on L2 size: on GB10 (24 MB L2) the 84 MB A operand streams from DRAM
once per N tile. On an RTX PRO 6000 (128 MB L2) it would mostly hit L2, so the penalty there is the 13 GB of L2→SM traffic
and the low arithmetic intensity of a 128×16 tile, which is smaller but probably not zero. Every variant gives bit-identical
output.

## Suggested fix

Rather than changing the constants for everyone, expose the tile and grid order to the existing tuner: add `tile_n` ∈ {16, 64,
128}, `stages` and `raster` ∈ {M-fastest, N-fastest} to `MtpFeedbackConfig`, which today tunes only the norm parameters.
Each GPU then measures its own best choice. On GB10 that is 128 / 2 / N-fastest; a 128-wide tile with 2 stages fits
comfortably in 99 KB on any SM12x. Our patch makes all three configurable via env and adds them to the compile-cache key:
<https://github.com/swesty/gb10-kernel-lab/blob/main/patches/b12x/mtp_feedback.tile-raster.diff>; harness:
<https://github.com/swesty/gb10-kernel-lab/blob/main/scripts/mtp_gemm_harness.py>. Happy to open a PR with fixed defaults
instead of env switches if preferred.

Environment: b12x 1.5.0 (`a321f9a6`), vLLM `local-inference-lab/vllm` `ab86b707`, image
`eugr/spark-vllm-b12x@sha256:66147fc1…a41b`, CUDA 13.0, CUTLASS DSL 4.7.0. Write-up: §13.15 of
<https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

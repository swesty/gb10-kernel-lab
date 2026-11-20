<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# MTP feedback prefill projection re-reads its A operand once per 16-column tile (22× slower than needed on GB10)

**Where:** `sequence/mtp_feedback/_cute_prefill.py` (`MTPPrefillBf16GemmKernel`; `_TILE_N` at line 29, `block_idx` at
line 280), FlashInfer `main` @ `d3f59f0` under `flashinfer/experimental/b12x/`. The same code is in the archived
standalone b12x 1.5.0 and in the `local-inference-lab/flashinfer` fork at `9873d7a`, which eugr's `spark-vllm-b12x` images build.

While profiling Qwen3.8-Flash-Next-NVFP4 on a DGX Spark (GB10, sm_121), I found one kernel taking **61 ms of every
4,096-token prefill chunk**, about 230 ms of a 16K-token prefill. It is the hidden projection of `mtp_feedback`, and it
was running at 3.5 TFLOPS on a GPU whose measured BF16 `mma.sync` peak is 124 TFLOPS.

## Why it is slow

The projection is `[tokens × 4, 2560] × [2560, 2560]ᵀ` plus a broadcast add of the token path. For one chunk that is
16,384 × 2,560 × 2,560, or 215 GFLOP. The kernel hard-codes `_TILE_N = 16` with `tile_m = 128` and launches
`grid = (M tiles, N tiles)`, so the blocks that share an A row panel are spread across the whole launch. The A operand is
84 MB of BF16, which does not fit in GB10's 24 MB L2, so it is streamed from memory once per N tile: 160 × 84 MB is about
13.4 GB per call. At roughly 220 GB/s, that is the ~60 ms I measured.

## Measurements on GB10

Standalone harness, M = 16,384, output checked against a PyTorch BF16 reference:

| Tile N | Stages | Grid order | Time | TFLOPS |
|---|---|---|---|---|
| 16 (current) | 3 | M fastest | 57,728 µs | 3.7 |
| 16 | 3 | N fastest | 8,419 µs | 25.5 |
| 64 | 3 | M fastest | 15,240 µs | 14.1 |
| 64 | 3 | N fastest | 3,223 µs | 66.6 |
| 128 | 3 | N fastest | 2,646 µs | 81.1 |
| **128** | **2** | **N fastest** | **2,619 µs (22×)** | **82.0** |
| cuBLAS BF16 matmul, same shape, no epilogue | | | 2,150–2,340 µs | 92–100 |

Every variant's output is bit-identical to the BF16 reference. In the serving engine, with tile N 128, 2 stages and
N-fastest order, a 14.5K-token real-text prefill went from 5.47 s to 5.27 s (−3.7%) and 16K TTFT from 5,842 ms to
5,639 ms (−3.5%). Decode did not change.

## It also helps sm_120

How much the narrow tile costs depends on L2 size, so I re-ran the same harness and the same b12x 1.5.0 source on an
RTX PRO 6000 (sm_120, 188 SMs, 128 MB L2). There the A operand mostly stays in L2, but the 16-wide tile still costs a
lot. 4,096 tokens, µs:

| Tile N / stages / grid order | RTX PRO 6000, back-to-back | RTX PRO 6000, after a 512 MB L2 flush |
|---|---|---|
| 16 / 3 / M fastest (current) | 1,764 | 1,919 |
| 16 / 3 / N fastest | 1,971 | 1,834 |
| 64 / 3 / M fastest | 770 | 856 |
| 128 / 2 / M fastest | **676** | 719 |
| 128 / 2 / N fastest (my GB10 setting) | 684 | **688** |
| cuBLAS BF16 matmul, no epilogue | 526 | 543 |

The 128-wide tile is a 2.6–2.8× win on sm_120 as well, again bit-identical. The grid order matters little once the tile
is 128, and on its own (at tile 16) N-fastest is 12% slower back-to-back on sm_120. So the tile is a clear fix, and the
order is a hardware- and cache-dependent choice.

## Suggested fix

Change the default tile to 128 wide with 2 stages. That fits comfortably in the 99 KB of shared memory on any SM12x part
and was faster on both GPUs I measured. Because the best grid order differs between GPUs, it would be better still to
let the existing tuner choose it: `MtpFeedbackConfig` tunes only the norm parameters today, and could also take
`tile_n` ∈ {16, 64, 128}, `stages` and the grid order.

My patch makes all three configurable through environment variables and adds them to the compile-cache key:
<https://github.com/swesty/gb10-kernel-lab/blob/main/patches/b12x/mtp_feedback.tile-raster.diff>. The harness is
<https://github.com/swesty/gb10-kernel-lab/blob/main/scripts/mtp_gemm_harness.py>. Happy to turn it into a PR with
fixed defaults instead of environment switches if you prefer.

Environment: b12x 1.5.0 (`a321f9a6`), vLLM `local-inference-lab/vllm` `ab86b707`, image
`eugr/spark-vllm-b12x@sha256:66147fc1…a41b`, CUDA 13.0, CUTLASS DSL 4.7.0 (4.7.1 on the RTX PRO 6000). Full write-up:
§13.15 and §13.21 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

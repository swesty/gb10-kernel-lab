<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# Dense GEMM: the persistent tile walk always goes along M, and quantized mode exposes no tile choice

**Where:** `_lib/dense_gemm.py`, `DenseGemmKernel._compute_grid` (line 5054), and `gemm/blockscaled/_preparation.py`
(`_lower_dense_with_hint`), FlashInfer `main` @ `d3f59f0`. The same code is in the archived standalone b12x 1.5.0.

These came out of profiling the MXFP8 dense projections of Qwen3.8-Flash-Next at prefill size (M = 4,096) on a DGX Spark
(GB10: 48 SMs, 24 MB L2, measured FP8 `mma.sync` peak 249 TFLOPS). Both are tuning choices whose best value depends on the
GPU, so I am not asking for new defaults. I measured them on GB10 and on an RTX PRO 6000 (sm_120), and the results below
suggest letting the tuner choose.

## 1. Tile order

`_compute_grid` always builds `PersistentTileSchedulerParams` with the default `raster_along_m=True`. For the GDN
`out_proj` (N = 2,560, K = 6,144) on GB10, the 25 MB A operand is re-read from memory once per column of N tiles. Nsight
Compute shows 250 MB read from memory against an ideal of about 41 MB, the tensor pipe 48% active, and TMA waits as the
top stall. Run time also depends on where A and W happen to be allocated: the same kernel on fresh copies of the same data
takes anywhere from 1.25 to 2.05 ms.

Walking along N when there are few N-tile columns (my patch uses `raster_along_m=False` for launches with 20 of them):

| GEMM (N × K), GB10 | M-major | N-major |
|---|---|---|
| out_proj 2560 × 6144 | 1,161–1,272 µs | 894–897 µs (−23 to −30%), no placement spread |
| shared-expert down 2560 × 640 | 119–131 µs | 107–115 µs |
| shared-expert gate+up 1280 × 2560 (A fits in L2) | 177 µs | 186 µs (worse, so excluded) |

The output is identical, since only the tile order changes. In the serving engine this gave −1.2% on a real-text prefill
and −1.7% on 16K TTFT.

On the RTX PRO 6000 (188 SMs, 128 MB L2), the picture is different. Same harness, a 512 MB L2 flush before each launch,
median of six placements, using the tiles sm_120's own lowering picks:

| GEMM (N × K), sm_120 default tile | M-major (current) | N-major |
|---|---|---|
| out_proj 2560 × 6144, 128×64 | 235.6 µs | 237.6 µs (+0.8%) |
| in_proj_qkvz 16384 × 2560, 128×128×64 | 528.4 µs | 534.5 µs (+1.1%) |
| attention qkv 13312 × 2560, 128×128×64 | 469.0 µs | 438.3 µs (−6.5%) |
| shared-expert gate+up 1280 × 2560, 128×64 | 65.6 µs | 63.5 µs (−3.2%) |
| shared-expert down 2560 × 640, 64×128 | 34.8 µs | 32.8 µs (−5.7%) |

The placement sensitivity also disappears there (all six placements within 1%). So the tile order is worth exposing, but
not worth hard-coding. Adding `raster_along_m` to the tuner's candidates, or applying N-major only under the existing
low-SM policy (`_use_low_sm_dense_tactics`, 64 SMs or fewer), would cover GB10 without touching larger parts.

## 2. No tile choice in quantized mode

`BlockscaledQuery` in `activation_mode="quantized"` lowers through `_lower_dense_with_hint` to a `DenseGemmQuery` with no
`overrides`, so the tuner never sees tiles outside the default lowering. On GB10 that lowering picks 64×128×128 for every
shape. Passing `overrides={"mma_tiler_mn": (128, 128), "_tile_k_override": 64, "_large_m_unroll_override": True}` for
the large-M plans gave:

- `in_proj_qkvz` (16384 × 2560): 2,150 µs to 1,926–2,074 µs
- attention qkv (13312 × 2560): 1,749 µs to 1,585 µs
- −0.5% end to end

Interestingly, 128×128×64 is the tile sm_120's lowering already uses for these shapes. One option is for the low-SM policy
to do the same for large M. The large-M unroll, though, cuts both ways on sm_120: `in_proj_qkvz` goes from 528 to 543 µs
(+2.7%) while attention qkv goes from 469 to 444 µs (−5.2%). That is why I would rather see the tuner explore tiles and
unroll for quantized-mode plans than have anyone adopt my overrides as-is.

## Related

The separate MXFP8 activation `_quantize` pass before each prefill GEMM costs 0.13–0.37 ms per call, and
`dense_gemm_fused_quant_a` only supports M ≤ 8. I fused the quantization into the producing kernels on the vLLM side
instead (through a small workspace hook in `_PackedExecutionState.run`), for −1.5% prefill (−2.0% TTFT). An in-GEMM path for large M would
make that unnecessary.

Patches: <https://github.com/swesty/gb10-kernel-lab/tree/main/patches/b12x> (`dense_gemm.raster-n.diff`,
`blockscaled_preparation.tile-override-and-prequant-hook.diff`). Harness and sweep data: `scripts/dense_gemm_*`,
`data/gemm/` and `data/sm120/`. Write-up: §13.16, §13.18 and §13.21 of the REPORT in the same repository.

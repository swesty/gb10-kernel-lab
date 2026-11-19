<!-- Draft issue for local-inference-lab/b12x. Not filed. -->
# Dense GEMM: M-major persistent tile walk re-streams A from DRAM on narrow outputs; quantized mode has no tile knob

**Where:** `b12x/_lib/dense_gemm.py` `DenseGemmKernel._compute_grid` and `gemm/blockscaled/_preparation.py`
(`_lower_dense_with_hint`); unchanged on `master` @ `2cc7f66`.

Measured on a DGX Spark (GB10, 48 SMs, 24 MB L2, measured FP8 `mma.sync` peak 249 TFLOPS) with the MXFP8 dense
projections of Qwen3.8-Flash-Next at prefill size (M = 4,096).

## 1. Tile order

`_compute_grid` always builds `PersistentTileSchedulerParams` with the default `raster_along_m=True`. For GDN `out_proj`
(N = 2,560, K = 6,144) the 25 MB A operand is re-read from DRAM once per column of N tiles. Nsight Compute: 250 MB read
from memory against an ideal ~41 MB, tensor pipe 48% active, top stall waiting on TMA. The run time also depends on where
A and W happen to be allocated: the same kernel on fresh copies of the same data takes 1.25–2.05 ms.

Walking along N when there are few N tiles (patch: `raster_along_m=False` for launches with 20 N-tile columns):

| GEMM (N × K) | M-major | N-major |
|---|---|---|
| out_proj 2560 × 6144 | 1,161–1,272 µs | 894–897 µs (−23 to −30%), no placement spread |
| shared-expert down 2560 × 640 | 119–131 µs | 107–115 µs |
| shared-expert gate+up 1280 × 2560 (A fits in L2) | 177 µs | 186 µs (worse, so excluded) |

Output is identical (only the tile order changes). Live: −1.2% real-text prefill, −1.7% 16K TTFT. A simple rule that
fits these shapes is "walk along N when the A operand is larger than W and than L2".

## 2. No tile control in quantized mode

`BlockscaledQuery` in `activation_mode="quantized"` lowers through `_lower_dense_with_hint` → `DenseGemmQuery` with no
`overrides`, so the tuner cannot pick tiles outside its default lowering. Passing
`overrides={"mma_tiler_mn": (128, 128), "_tile_k_override": 64, "_large_m_unroll_override": True}` for large-M plans:
`in_proj_qkvz` (16384 × 2560) 2,150 → 1,926–2,074 µs, attention qkv (13312 × 2560) 1,749 → 1,585 µs; live −0.5%.

## Related

The separate MXFP8 activation `_quantize` pass before each prefill GEMM costs 0.13–0.37 ms per call.
`dense_gemm_fused_quant_a` only supports M ≤ 8. We fused the quantisation into the producing kernels instead (vLLM side,
through a small workspace hook in `_PackedExecutionState.run`), for −2% prefill. An in-GEMM path for large M would make
that unnecessary.

Patches: <https://github.com/swesty/gb10-kernel-lab/tree/main/patches/b12x> (`dense_gemm.raster-n.diff`,
`blockscaled_preparation.tile-override-and-prequant-hook.diff`); harness and sweep data: `scripts/dense_gemm_*`,
`data/gemm/`. Write-up: §13.16 and §13.19 of the REPORT.

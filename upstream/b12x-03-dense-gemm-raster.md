<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x) and local-inference-lab/b12x. Not filed. -->
# Dense GEMM: M-major persistent tile walk re-streams A from DRAM on narrow outputs; quantized mode has no tile knob

**Where:** `_lib/dense_gemm.py` `DenseGemmKernel._compute_grid` (FlashInfer line 5054) and `gemm/blockscaled/_preparation.py`
(`_lower_dense_with_hint`) — present in FlashInfer `main` @ `d3f59f0` and local-inference-lab/b12x `master` @ `2cc7f66`.

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

Output is identical (only the tile order changes). Live: −1.2% real-text prefill, −1.7% 16K TTFT.

**This is hardware-dependent, so we are not proposing a new default.** We re-ran the same harness on an RTX PRO 6000
(sm_120, 188 SMs, 128 MB L2; 512 MB L2 flush before each launch; median of 6 placements, µs). sm_120's own lowering
already picks different tiles (128×128×64 for the big projections, 128×64 or 64×128 elsewhere; GB10's low-SM lowering picks
64×128×128 everywhere):

| GEMM (N × K), sm_120 default tile | M-major (stock) | N-major |
|---|---|---|
| out_proj 2560 × 6144, 128×64 | 235.6 | 237.6 (+0.8%) |
| in_proj_qkvz 16384 × 2560, 128×128×64 | 528.4 | 534.5 (+1.1%) |
| attention qkv 13312 × 2560, 128×128×64 | 469.0 | 438.3 (−6.5%) |
| shared-expert gate+up 1280 × 2560, 128×64 | 65.6 | 63.5 (−3.2%) |
| shared-expert down 2560 × 640, 64×128 | 34.8 | 32.8 (−5.7%) |

Placement sensitivity is a GB10 effect: on sm_120 the six placements are within 1% (GB10: 1.25–2.05 ms for out_proj).
We propose adding `raster_along_m` to the tuner's candidate set, or applying N-major only under the existing low-SM / Spark
policy (`_use_low_sm_dense_tactics`, ≤ 64 SMs).

## 2. No tile control in quantized mode

`BlockscaledQuery` in `activation_mode="quantized"` lowers through `_lower_dense_with_hint` → `DenseGemmQuery` with no
`overrides`, so the tuner cannot pick tiles outside its default lowering. Passing
`overrides={"mma_tiler_mn": (128, 128), "_tile_k_override": 64, "_large_m_unroll_override": True}` for large-M plans:
`in_proj_qkvz` (16384 × 2560) 2,150 → 1,926–2,074 µs, attention qkv (13312 × 2560) 1,749 → 1,585 µs; live −0.5% (sm_121).
On GB10 this moves the large-M plans to the tile sm_120's lowering already uses (128×128×64), so one option is for the low-SM
policy to do the same for large M. Large-M unroll is shape-dependent on sm_120 (in_proj_qkvz 528 → 543 µs, +2.7%; attention qkv
469 → 444 µs, −5.2%), so the request is to let the tuner explore tiles and unroll for quantized-mode plans, not to hard-code
ours.

## Related

The separate MXFP8 activation `_quantize` pass before each prefill GEMM costs 0.13–0.37 ms per call.
`dense_gemm_fused_quant_a` only supports M ≤ 8. We fused the quantisation into the producing kernels instead (vLLM side,
through a small workspace hook in `_PackedExecutionState.run`), for −2% prefill. An in-GEMM path for large M would make
that unnecessary.

Patches: <https://github.com/swesty/gb10-kernel-lab/tree/main/patches/b12x> (`dense_gemm.raster-n.diff`,
`blockscaled_preparation.tile-override-and-prequant-hook.diff`); harness and sweep data: `scripts/dense_gemm_*`,
`data/gemm/`. Write-up: §13.16 and §13.19 of the REPORT.

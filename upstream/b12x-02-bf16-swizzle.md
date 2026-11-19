<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# BF16 shared-memory tiles: the 128-byte swizzle atom is applied to element offsets, giving 2-way bank conflicts

**Where** (present in FlashInfer `main` @ `d3f59f0` under `flashinfer/experimental/b12x/`; also in the archived standalone local-inference-lab/b12x `master` @ `2cc7f66` / b12x 1.5.0):
- `b12x/attention/paged/_selected_forward.py`: the Q, K, V and P tiles of the selected-position (QSA) paged attention kernel.
- `b12x/gemm/bf16_gemv/_kernel.py:389–390`: `shared_a` / `shared_b`.

## The pattern

Both build a BF16 smem layout from the stock 128-byte atom (`get_smem_layout_atom(ROW_MAJOR, BFloat16, k)` → `K_SW128`,
`S<3,4,3> ∘ (8,64):(64,1)`), wrap it as a `ComposedLayout`, and pass it straight to
`allocator.allocate_tensor(BFloat16, composed_layout)`. That applies `S<3,4,3>` to **element** offsets. For 2-byte
elements it XORs 32-byte units instead of 16-byte units, so pairs of rows land in the same banks.

The TMA-fed kernels use `get_tensor(layout.outer, swizzle=layout.inner)`, which applies the swizzle to the pointer
(byte addresses). They have to, because TMA writes byte-swizzled data. They are not affected.

## Evidence (selected-position attention, GB10 / sm_121, Nsight Compute)

- Every 128-bit store into the K/V tiles and every `ldmatrix` showed about 2× the ideal wavefronts (43% excess on
  stores, 54% on `ldmatrix`, 47% on the P stores), independent of the thread→data mapping. That points at the
  layout, not the access pattern.
- `S<3,3,3>` for Q/K/V (16-byte units) and the matching atom for P removes the load (`ldmatrix`) conflicts: 46.8% → 0.0% of
  load wavefronts. On its own it leaves the K/V gather stores at 51% conflicts, because of the order the gather writes 16-byte
  chunks in.
- Storing the odd 8-column chunk first in odd 64-column blocks (done in our pipelined gather path) brings the stores to
  4.6–5.0%.
- Together with two other changes (software-pipelined loads; QK and online softmax split across all four warps), the
  kernel went from 14.04 ms to 7.31 ms at 16K context (1.92×) with the same error against FP32 (1e-4, LSE exact).
  The swizzle alone was 8.68 → 7.77 ms on top of pipelining.

`bf16_gemv` has the same construction; its conflicts are expected but **not measured**. That kernel is not on the
Flash Next path.

## Same on sm_120 (RTX PRO 6000)

Bank conflicts depend only on the SM's shared-memory banks, which are the same on sm_120 and sm_121. We checked: same harness,
same b12x 1.5.0 source, Nsight Compute bank-conflict counters, kv_warps = 2, 16K context:

| Variant | GB10 loads | GB10 stores | RTX PRO 6000 loads | RTX PRO 6000 stores |
|---|---|---|---|---|
| stock | 46.8% | 51.6% | 46.8% | 51.7% |
| `S<3,3,3>` only | 0.0% | 51.1% | 0.0% | 51.2% |
| pipelined gather | 46.8% | 51.6% | 46.8% | 51.5% |
| pipelined gather + `S<3,3,3>` + store order | 0.0% | 5.0% | 0.0% | 4.7% |

(Conflicts as a share of shared-memory wavefronts.) Time on the RTX PRO 6000, 4,096 rows, µs, 16K / 32K context: stock
2,958 / 2,957; `S<3,3,3>` only 2,882 / 2,795 (−2.6% / −5.5%); pipelined 2,026 / 1,922; pipelined + swizzle 1,750 / 1,749
(−13.6% / −9.0%); all three changes 1,615 / 1,772 (1.83× / 1.67×). Error against FP32 is unchanged in every variant
(1e-4, LSE exact).

## Patch / references

- Kernel with the fix behind `B12X_SELECTED_SWZ333` (plus the other two changes behind their own switches):
  <https://github.com/swesty/gb10-kernel-lab/tree/main/harness/attention/patch>
- Harness and Nsight Compute reports, before and after: `harness/attention/`, `data/ncu/attention/` in the same repo.
- Write-up: §9.6–9.8, §11–12 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

A quick way to catch other instances: grep for `allocate_tensor(BFloat16,` / `allocate_tensor(Float16,` whose layout
comes from `make_smem_layout_atom` / `tile_to_shape` of a swizzled atom.

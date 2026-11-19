<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x) and local-inference-lab/b12x. Not filed. -->
# BF16 shared-memory tiles: the 128-byte swizzle atom is applied to element offsets, giving 2-way bank conflicts

**Where** (present in FlashInfer `main` @ `d3f59f0` under `flashinfer/experimental/b12x/`, and in local-inference-lab/b12x `master` @ `2cc7f66`):
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
- Using `S<3,3,3>` for Q/K/V (16-byte units), the matching atom for P, and storing the odd 8-column chunk first in odd
  64-column blocks gives **0% excess wavefronts on every shared-memory operation**.
- Together with two other changes (software-pipelined loads; QK and online softmax split across all four warps), the
  kernel went from 14.04 ms to 7.31 ms at 16K context (1.92×) with the same error against FP32 (1e-4, LSE exact).
  The swizzle alone was 8.68 → 7.77 ms on top of pipelining.

`bf16_gemv` has the same construction; its conflicts are expected but **not measured**. That kernel is not on the
Flash Next path.

Measured on sm_121 (GB10). This is not a tuning choice: bank conflicts depend only on the SM's shared-memory banks, which
are the same on sm_120 and sm_121. So the fix should remove the same conflicts on an RTX PRO 6000. We have not measured it
there.

## Patch / references

- Kernel with the fix behind `B12X_SELECTED_SWZ333` (plus the other two changes behind their own switches):
  <https://github.com/swesty/gb10-kernel-lab/tree/main/harness/attention/patch>
- Harness and Nsight Compute reports, before and after: `harness/attention/`, `data/ncu/attention/` in the same repo.
- Write-up: §9.6–9.8, §11–12 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

A quick way to catch other instances: grep for `allocate_tensor(BFloat16,` / `allocate_tensor(Float16,` whose layout
comes from `make_smem_layout_atom` / `tile_to_shape` of a swizzled atom.

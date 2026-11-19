<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# BF16 shared-memory tiles apply the 128-byte swizzle to element offsets, causing 2-way bank conflicts

**Where:** FlashInfer `main` @ `d3f59f0` under `flashinfer/experimental/b12x/` (the same code is in the archived
standalone b12x 1.5.0):

- `attention/paged/_selected_forward.py`: the Q, K, V and P tiles of the selected-position (QSA) paged attention kernel.
- `gemm/bf16_gemv/_kernel.py:389–390`: `shared_a` and `shared_b`.

I ran into this while profiling the selected-position attention kernel for Qwen3.8-Flash-Next on a DGX Spark (GB10,
sm_121). Nsight Compute showed every 128-bit store into the K/V tiles and every `ldmatrix` taking about twice the ideal
number of shared-memory wavefronts. Changing the thread-to-data mapping did not help, which pointed at the layout rather
than the access pattern.

## The pattern

Both sites build a BF16 shared-memory layout from the stock 128-byte atom (`get_smem_layout_atom(ROW_MAJOR, BFloat16, k)`,
which gives `K_SW128`, `S<3,4,3> ∘ (8,64):(64,1)`), wrap it as a `ComposedLayout`, and pass it straight to
`allocator.allocate_tensor(BFloat16, composed_layout)`. That applies `S<3,4,3>` to **element** offsets. For 2-byte
elements, the XOR then works on 32-byte units instead of 16-byte units, so pairs of rows land in the same banks.

The TMA-fed kernels are not affected. They use `get_tensor(layout.outer, swizzle=layout.inner)`, which applies the
swizzle to byte addresses. They have to, because TMA writes byte-swizzled data.

A quick way to find other instances: grep for `allocate_tensor(BFloat16,` or `allocate_tensor(Float16,` where the layout
comes from `make_smem_layout_atom` or `tile_to_shape` of a swizzled atom.

## What fixed it

- Using `S<3,3,3>` (16-byte units) for Q/K/V, and the matching atom for P, removes the load conflicts: from 46.8% of load
  wavefronts to 0.0%.
- On its own, that leaves the K/V gather stores at 51% conflicts, because of the order in which the gather writes its
  16-byte chunks. Storing the odd 8-column chunk first in odd 64-column blocks (which my pipelined gather path does)
  brings the stores down to 4.6–5.0%.
- Together with two unrelated changes (software-pipelined loads, and QK plus the online softmax split across all four
  warps), the kernel went from 14.04 ms to 7.31 ms at 16K context (1.92×), with the same error against FP32 (1e-4, LSE
  exact). The swizzle accounted for 8.68 to 7.77 ms of that, on top of pipelining.

I have not measured `bf16_gemv`. It has the same construction, so I expect the same conflicts, but it is not on the
Flash Next path.

## Same on sm_120

Bank conflicts depend only on the SM's shared-memory banks, which are the same on sm_120 and sm_121, so this should not be
GB10-specific. To confirm, I ran the same harness and the same b12x 1.5.0 source on an RTX PRO 6000 (sm_120) and read the
same Nsight Compute counters (kv_warps = 2, 16K context). Conflicts as a share of shared-memory wavefronts:

| Variant | GB10 loads | GB10 stores | RTX PRO 6000 loads | RTX PRO 6000 stores |
|---|---|---|---|---|
| current | 46.8% | 51.6% | 46.8% | 51.7% |
| `S<3,3,3>` only | 0.0% | 51.1% | 0.0% | 51.2% |
| pipelined gather | 46.8% | 51.6% | 46.8% | 51.5% |
| pipelined gather + `S<3,3,3>` + store order | 0.0% | 5.0% | 0.0% | 4.7% |

On the RTX PRO 6000 (4,096 rows, 16K / 32K context), the kernel takes 2,958 / 2,957 µs as it stands. With `S<3,3,3>`
only it takes 2,882 / 2,795 µs (−2.6% / −5.5%). Pipelined, it takes 2,026 / 1,922 µs, and pipelined plus the swizzle fix
1,750 / 1,749 µs (−13.6% / −9.0%). All three changes together give 1,615 / 1,772 µs (1.83× / 1.67×). Error against FP32
is unchanged in every variant.

## Patch and data

- The kernel with the fix behind `B12X_SELECTED_SWZ333` (and the two other changes behind their own switches):
  <https://github.com/swesty/gb10-kernel-lab/tree/main/harness/attention/patch>
- The harness and the Nsight Compute reports, before and after: `harness/attention/`, `data/ncu/attention/` and
  `data/sm120/` in the same repository.
- Write-up: §9.6–9.8, §11–12 and §13.21 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

Happy to send a PR for the attention kernel's layouts, and to measure `bf16_gemv` if that would help.

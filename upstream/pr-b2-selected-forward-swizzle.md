<!--
Draft PR for flashinfer-ai/flashinfer. Not opened.
Title: fix(experimental/b12x): swizzle selected-position attention tiles in 16-byte units
Base: main @ d3f59f0. Patch: pr-b2-selected-forward-swizzle.diff (applies cleanly).
Before opening: review every changed line yourself, run FlashInfer's pre-commit hooks in a real checkout,
and fill in the commit SHA the tests ran on. The body below follows the default template unchanged.
-->
<!-- .github/pull_request_template.md -->

## 📌 Description

The selected-position (QSA) paged attention kernel in
`flashinfer/experimental/b12x/attention/paged/_selected_forward.py` builds its Q, K, V and P shared-memory layouts from the
stock 128-byte atom (`get_smem_layout_atom(ROW_MAJOR, BFloat16, k)`, which is `S<3,4,3> ∘ (8,64):(64,1)`), then
allocates them with `allocator.allocate_tensor(BFloat16, layout)`. That call applies the swizzle to **element** offsets.
For 2-byte elements, `S<3,4,3>` then XORs 32-byte units instead of 16-byte units, so pairs of rows land in the same
banks. The TMA-fed kernels are not affected, because they use `get_tensor(layout.outer, swizzle=layout.inner)`, which
swizzles byte addresses.

I found it while profiling this kernel for Qwen3.8-Flash-Next on a DGX Spark: Nsight Compute showed every `ldmatrix` and
every 128-bit store into the K/V tiles taking about twice the ideal number of shared-memory wavefronts, and changing the
thread mapping did not help, which pointed at the layout.

This PR:

- uses `S<3,3,3> ∘ (8,64):(64,1)` (16-byte units) for the Q, K and V tiles;
- uses the matching `S<b,3,3> ∘ (8,tile_n):(tile_n,1)` for the P tile, with b = 1, 2 or 3 for `tile_n` = 16, 32 or 64
  (the only values `16 * kv_warps` can take);
- bumps the `attention.paged.selected_forward` compile-cache version from 4 to 5;
- drops three imports that are no longer used.

**Bank conflicts.** Nsight Compute counters, kv_warps = 2, 16K context, 4,096 rows, conflicts as a share of
shared-memory wavefronts. I measured these with the same layout change behind a switch in standalone b12x 1.5.0:

| | GB10 (sm_121) loads | GB10 stores | RTX PRO 6000 (sm_120) loads | RTX PRO 6000 stores |
|---|---|---|---|---|
| before | 46.8% | 51.6% | 46.8% | 51.7% |
| after | 0.0% | 51.1% | 0.0% | 51.2% |

The counts match to three significant figures on both GPUs, as expected from identical bank layouts. The K/V gather
stores stay conflicted, because of the order in which the gather writes its 16-byte chunks. A store-order change (odd
8-column chunk first in odd 64-column blocks) brings them to about 5%, but in my patch it lives in a software-pipelined
gather path, so I would rather send it separately with that work.

**Kernel time.** µs, 4,096 rows, kv_warps = 1 / 2 / 4, with this PR's file:

| Context | GB10, before | GB10, after | RTX PRO 6000, before | RTX PRO 6000, after |
|---|---|---|---|---|
| 16K | 17,624 / 14,080 / 24,480 | 17,661 / **13,414** / 21,480 | 3,604 / 3,054 / 4,960 | 3,455 / **2,857** / 4,679 |
| 32K | 17,527 / 13,956 / 24,554 | 17,479 / **13,282** / 21,470 | 3,506 / 2,893 / 4,800 | 3,450 / 2,874 / 4,762 |

At kv_warps = 2 (what the QSA path uses) this is −4.7% and −4.8% on GB10. On the RTX PRO 6000 the run shared the GPU
with an idle served model, so treat those numbers as noisy. An earlier run of the same layout change with the GPU to
itself measured 2,958 to 2,882 µs at 16K (−2.6%) and 2,957 to 2,795 µs at 32K (−5.5%). kv_warps = 4 gains 12% on GB10
and 1–6% on the RTX PRO 6000; kv_warps = 1 is unchanged on GB10 and within the noise of that shared run on the RTX PRO
6000.

**Correctness.** Maximum absolute error against an FP32 reference is unchanged (1e-4), and the LSE matches exactly, on
both GPUs.

**How I measured.** RTX PRO 6000: FlashInfer 0.7.3 from the `vllm/vllm-openai:v0.34.0` image (its
`_selected_forward.py` is byte-identical to main `d3f59f0`) with this file dropped in. GB10: a DGX Spark running
standalone b12x 1.5.0 (whose `_selected_forward.py` differs from main only in formatting) with this file dropped in,
beside a running vLLM engine. Timing used CUDA events in a standalone harness that calls
`launch_selected_paged_gqa_direct` at the Qwen3.8 shapes (24/2 heads, head dim 256, FP8 KV, 3,024-token pages). Harness,
Nsight Compute reports, raw output and write-up: https://github.com/swesty/gb10-kernel-lab (`harness/attention/`,
`data/sm120/`, `data/pr-validation-20261119/`, REPORT §9.7 and §13.21).

I found and fixed this with an AI coding agent (Claude Code) doing most of the profiling and patch work; I reviewed the
change and can walk through it. Thanks to the b12x authors for the kernels.

## 🔍 Related Issues

None found. Part of the experimental b12x integration tracked in #6909. The same `allocate_tensor(BFloat16,
composed_layout)` construction appears in `gemm/bf16_gemv/_kernel.py:389–390`; I have not measured it and did not
change it here.

## 🚀 Pull Request Checklist

Thank you for contributing to FlashInfer! Before we review your pull request, please make sure the following items are complete.

### ✅ Pre-commit Checks

- [ ] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [ ] I have installed the hooks with `pre-commit install`.
- [ ] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure about how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

## 🧪 Tests

- [ ] Tests have been added or updated as needed.
- [x] All tests are passing (`unittest`, etc.).
- [ ] This PR requires **full-scope** CI, not targeted tests (expensive; justify below).
      Reason: <!-- e.g. cross-cutting change to shared infrastructure -->

  Suggested CI test targets (optional; paths under `tests/`, whitespace-separated — a starting
  point for `@flashinfer-bot run <targets>` / `/bot run <targets>`, reviewers may adjust): `tests/experimental/b12x/attention/test_qsa_sparse_gqa.py`

`tests/experimental/b12x/attention/test_qsa_sparse_gqa.py` passes in full on the RTX PRO 6000 with and without this
change (77 passed). It is the only test file that exercises this kernel, and it already checks the output against a
reference, so I did not add a test: the change is a layout permutation with no new code path. I ran the GB10 check
through the harness rather than pytest.

## 🔬 Experimental Track

<!-- Only for PRs submitted under the experimental policy (CONTRIBUTING.md → "Experimental APIs and Backends").
     Leave this section untouched for normal PRs. -->

- [x] This PR is **experimental**: it adds or changes code under `flashinfer/experimental/` and/or an `@flashinfer_experimental_api`. Tracking issue: #6909
  - [ ] The tracking issue names an owner, the reason for the experimental path, and a graduation plan with a target release.
  - [x] Core changes are limited to a thin entry point (signature, shared validation, feature-gate check, backend selection, handoff).
  - [ ] Tests live in `tests/experimental/` and were validated on the intended hardware; a runnable example is included. (Existing tests, validated on sm_120; no example added.)
  - [x] Nothing is registered in `flashinfer/aot.py`, and no experimental backend is reachable from `backend="auto"` without `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. (Calling an `@flashinfer_experimental_api` or naming a backend explicitly is itself the opt-in and needs no environment variable.)
  - [x] **Test scope declared below.** The experimental CI lane runs exactly these targets, so keep them as narrow as the change allows.

```experimental-tests
tests/experimental/b12x/attention/test_qsa_sparse_gqa.py
```

## Reviewer Notes

- This is a small fix to an existing experimental backend, so I linked the existing tracking issue. There are no core
  changes.
- A quick way to find other instances of the pattern: grep for `allocate_tensor(BFloat16,` or `allocate_tensor(Float16,`
  where the layout comes from `make_smem_layout_atom` or `tile_to_shape` of a swizzled atom.
- Follow-ups I can send: the store-order change that removes most of the remaining store conflicts, the software
  pipelining and split-QK changes (together 1.83× at 16K on the RTX PRO 6000), and a measurement of `bf16_gemv`.
- `_selected_forward.py` is byte-identical in the `local-inference-lab/flashinfer` fork at `9873d7a`, which eugr's
  `spark-vllm-b12x` images build, so this patch also applies there unchanged.

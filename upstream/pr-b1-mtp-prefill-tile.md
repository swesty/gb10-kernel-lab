<!--
Draft PR for flashinfer-ai/flashinfer. Not opened.
Title: perf(experimental/b12x): 128-wide N tile, N-fastest grid and proxy fence in the MTP feedback prefill GEMM
Base: main @ 560d7e2 (2026-11-23). Patch: pr-b1-mtp-prefill-tile.diff (applies cleanly; adds one test; passes
ruff 0.12.8 format and check, the version FlashInfer's pre-commit pins). The two files it touches are byte-identical
at 560d7e2, v0.7.6rc1 (691183f) and d3f59f0.
Before opening: review every changed line yourself, run FlashInfer's pre-commit hooks in a real checkout,
and fill in the commit SHA the tests ran on. The body below follows the default template unchanged.
-->
<!-- .github/pull_request_template.md -->

## 📌 Description

The MTP feedback prefill projection (`MTPPrefillBf16GemmKernel` in
`flashinfer/experimental/b12x/sequence/mtp_feedback/_cute_prefill.py`) hard-codes `_TILE_N = 16` and launches
`grid = (M tiles, N tiles)`. For the Qwen3.8 shape (rows = 4 × tokens, 2,560 × 2,560), every block re-reads its
128-row A panel once per 16-wide N tile, and blocks that share a panel are spread across the whole launch. At 4,096
tokens the A operand is 84 MB of BF16, so on a DGX Spark (GB10, 24 MB L2) it streams from DRAM 160 times, about 13.4 GB
per call. That one GEMM took 57.6 ms per 4,096-token chunk, against 2.2 ms for cuBLAS on the same shape.

This PR:

- picks the N tile per kernel instance: 128 when `output_columns` is a multiple of 128 (the only Qwen3.8 case), else
  64, else 16, so every width the kernel accepted before still compiles;
- uses 2 stages at tile 128 (3 at 64 and 16), which keeps shared memory well under 99 KB;
- puts the N tiles on grid x, so blocks that share an A panel run back to back and reuse it from L2;
- adds a generic-to-async proxy fence before each stage release in the mainloop, the same fix and the same CUDA 13.4
  gate as #9759 (see "Proxy fence" below);
- bumps the `sequence.mtp_feedback.prefill_bf16_gemm` compile-cache version from 3 to 4, so binaries compiled with the
  old tile or without the fence are not reused;
- adds `test_prefill_gemm_tile_n_choice_matches_reference`, which checks each tile width (128, 64 and 16) against a
  PyTorch reference with a partial live-row count.

**Results.** Kernel time in µs, before and after, rows = 4 × tokens, 2,560 × 2,560, median of 20 individually timed
launches, each after an L2 flush (96 MB on GB10, 512 MB on the RTX PRO 6000):

| Tokens | GB10 (sm_121), before | GB10, after | RTX PRO 6000 (sm_120), before | RTX PRO 6000, after |
|---|---|---|---|---|
| 16 | 123 | 143 | 31 | 39 |
| 64 | 158 | 150 | 47 | 49 |
| 256 | 554 | 308 | 133 | 59 |
| 1,024 | 2,186 | 799 | 473 | 207 |
| 4,096 | 57,640 | **2,724** | 1,917 | **696** |
| cuBLAS BF16, 4,096 tokens, no epilogue | 2,221 | | 549 | |

At 256 tokens and up the new tile is 1.8–21× faster on GB10 and 2.2–2.8× faster on the RTX PRO 6000. At 16 tokens it
costs 8–20 µs per call, because 20 wide tiles leave most SMs idle where 160 narrow ones did not. At 64 tokens it is about
even: 8 µs faster on GB10 and 2 µs slower on the RTX PRO 6000. I kept one default rather
than dispatching on the live row count, because the kernel is compiled per capacity and the cost is small next to the
saving; happy to add a small-M path if you would rather have it.

**Why N-fastest.** On GB10 the grid order matters once the tile is wide: at 4,096 tokens, 128-wide with M-fastest
order takes 8,232 µs against 2,703 µs with N-fastest. The exception is 256 tokens (258 vs 307 µs, M-fastest better).
On the RTX PRO 6000 the two orders are within 5% at every size, so N-fastest costs nothing there.

**End to end.** In vLLM serving Qwen3.8-Flash-Next-NVFP4 on a DGX Spark, the same change (applied to standalone b12x
1.5.0 behind a switch) took a 14.5K-token prefill from 5.47 s to 5.27 s (−3.7%) and 16K TTFT from 5,842 ms to 5,639 ms
(−3.5%). Decode did not change.

**Correctness.** At 256 tokens and up, the output is 100% bit-identical to a BF16 PyTorch reference, before and after.
At 16 and 64 tokens, the patched kernel matches the reference on exactly the same fraction of elements as the current
kernel (the reference there is a different cuBLAS algorithm).

**Proxy fence.** The table above is the tile and grid change alone; the fence came later and its cost is measured
separately below. In the mainloop, each stage is read with `ldmatrix` (the generic proxy) and refilled by TMA (the async
proxy), and the consumer hands it back with a plain mbarrier arrive. That arrive doesn't order the `ldmatrix` reads
before the TMA write, which is the hazard #9759 fixes in the b12x dense GEMM and #7785 fixed in the GDN delta-rule
kernel after an overwrite was observed there. #9759, #9760 and #9761 don't touch this file. In the current kernel's SASS
(b12x 1.5.0, cutlass-dsl 4.7.0 with `cutlass.CUDA_VERSION` 13.3, sm_121a, K = 2,560), all 40 consumer releases are
issued while some of that k-block's `LDSM` results are still unread, and none is fenced. That holds for the 16-wide
tile and for this PR's 128-wide tile. Since this PR already changes the pipeline depth in that loop, it adds the fence
#9759 uses, behind the same `(13, 4)` gate. The gate is needed: on CUDA 13.3 the ungated
`fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster` fails in libNVVM ("NVVM backend
compilation failed"), so older toolkits get `cute.arch.fence_proxy("async.shared", space="cta")`.

With 13.3, ptxas puts `MEMBAR.ALL.CTA` then `FENCE.VIEW.ASYNC.S` after the last `LDSM` of every k-block, seven HMMAs
before the arrive. I checked that in all 12 MTP GEMM kernels the GB10 run compiled (harness sizes plus the test file):
40 of 40 releases fenced in each, against 0 of 40 without the patch. Registers and shared memory don't change (164 and
220 registers per thread at 4,096 and 16 tokens, zero local loads and stores). Kernel time on GB10 at tile 128, two
alternating runs per row, each the median of 20 launches after a 96 MB L2 flush:

| Tokens | Without fence (µs) | With fence (µs) | Change |
|---|---|---|---|
| 16 | 130, 127 | 128, 125 | none (within noise) |
| 64 | 141, 139 | 141, 141 | +0.7% |
| 256 | 294, 292 | 292, 297 | +0.5% |
| 1,024 | 787, 790 | 799, 796 | +1.1% |
| 4,096 | 2,718, 2,717 | 2,732, 2,785 | +1.5% |

A separate A/B in the serving container (three alternating runs per row) gave +1.0% back to back and +0.9% after an L2
flush at 4,096 tokens, and +0.4% / −0.4% for the old 16-wide tile. At 4,096 tokens that's about 40 µs per prefill chunk.
Output matches the BF16 reference bit for bit with and without the fence. I haven't seen corruption without it either
(0 of 500 repeated launches differed from the first), so this is a memory-model fix rather than a fix for a failure I
reproduced. On CUDA 13.4 and later the fence should compile to nothing, but I have no 13.4 toolkit here to confirm
that, and I haven't re-run the fence on the RTX PRO 6000.

**How I measured.** GB10: a DGX Spark running standalone b12x 1.5.0 (whose `_cute_prefill.py` differs from main only
in formatting) with this file dropped in, cutlass-dsl 4.7.0, torch 2.13 cu130, beside a running vLLM engine. RTX PRO
6000: FlashInfer 0.7.3 from the `vllm/vllm-openai:v0.34.0` image (its `_cute_prefill.py` is byte-identical to main
`d3f59f0`) with this file dropped in, cutlass-dsl 4.8.0, torch 2.14, beside an idle served model. Timing used CUDA
events, not `bench_gpu_time`; I can re-run with `flashinfer.testing.bench_gpu_time(..., enable_cupti=True)` if you
prefer. Harness, raw output and full write-up:
https://github.com/swesty/gb10-kernel-lab (`scripts/mtp_gemm_harness.py`, `data/pr-validation-20261119/`,
`data/pr-validation-20261123-fence/`, `data/sm120/`, REPORT §13.15 and §13.21).

I found and fixed this with an AI coding agent (Claude Code) doing most of the profiling and patch work; I reviewed the
change and can walk through it. Thanks to the b12x authors for the kernels.

## 🔍 Related Issues

None found for the tile and grid order. Part of the experimental b12x integration tracked in #6909. The fence follows
#9759 (with #9760 and #9761 for the MoE and attention kernels), which doesn't cover this file. If #9759 lands first
with a shared helper, I'll switch this call site to it.

## 🚀 Pull Request Checklist

Thank you for contributing to FlashInfer! Before we review your pull request, please make sure the following items are complete.

### ✅ Pre-commit Checks

- [ ] I have installed `pre-commit` by running `pip install pre-commit` (or used your preferred method).
- [ ] I have installed the hooks with `pre-commit install`.
- [ ] I have run the hooks manually with `pre-commit run --all-files` and fixed any reported issues.

> If you are unsure about how to set up `pre-commit`, see [the pre-commit documentation](https://pre-commit.com/).

## 🧪 Tests

- [x] Tests have been added or updated as needed.
- [ ] All tests are passing (`unittest`, etc.).
- [ ] This PR requires **full-scope** CI, not targeted tests (expensive; justify below).
      Reason: <!-- e.g. cross-cutting change to shared infrastructure -->

  Suggested CI test targets (optional; paths under `tests/`, whitespace-separated — a starting
  point for `@flashinfer-bot run <targets>` / `/bot run <targets>`, reviewers may adjust): `tests/experimental/b12x/sequence/test_mtp_feedback.py`

The new test passes on both GPUs (3 of 3 on sm_121 and sm_120). On FlashInfer 0.7.3, whose `test_mtp_feedback.py` is
identical to main's, 8 existing tests in that file fail with `AttributeError: 'Binding' object has no attribute
'execution'` both with and without this change (no source file in `mtp_feedback/` defines `execution`). With this change
the file gives 12 passed, 8 failed, 1 skipped on both GPUs, against 9 passed, 8 failed, 1 skipped before: the same 8
failures by name, plus the 3 new tests (logs in `data/pr-validation-20261119/*/pytest_mtp_*.txt`). With the fence
added I re-ran the file on GB10 and got the same result: 12 passed, the same 8 failures by name, 1 skipped
(`data/pr-validation-20261123-fence/gb10-spark5/pytest_fenced.txt`). I haven't re-run it with the fence on sm_120. I
can fix those 8 tests in a separate PR.

## 🔬 Experimental Track

<!-- Only for PRs submitted under the experimental policy (CONTRIBUTING.md → "Experimental APIs and Backends").
     Leave this section untouched for normal PRs. -->

- [x] This PR is **experimental**: it adds or changes code under `flashinfer/experimental/` and/or an `@flashinfer_experimental_api`. Tracking issue: #6909
  - [ ] The tracking issue names an owner, the reason for the experimental path, and a graduation plan with a target release.
  - [x] Core changes are limited to a thin entry point (signature, shared validation, feature-gate check, backend selection, handoff).
  - [ ] Tests live in `tests/experimental/` and were validated on the intended hardware; a runnable example is included. (Tests: yes, on sm_121 and sm_120. Example: none added; see Reviewer Notes.)
  - [x] Nothing is registered in `flashinfer/aot.py`, and no experimental backend is reachable from `backend="auto"` without `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1`. (Calling an `@flashinfer_experimental_api` or naming a backend explicitly is itself the opt-in and needs no environment variable.)
  - [x] **Test scope declared below.** The experimental CI lane runs exactly these targets, so keep them as narrow as the change allows.

```experimental-tests
tests/experimental/b12x/sequence/test_mtp_feedback.py
```

## Reviewer Notes

- This is a small improvement to an existing experimental backend, so I linked the existing tracking issue rather than
  opening a new one. There are no core changes, and no new example (the existing MTP feedback path exercises it).
- The wider tile raises register use, so I checked for spills with Nsight Compute on both GPUs. At 4,096 tokens
  (`tile_m` 128) the kernel uses 164 registers per thread, and at 16 tokens (`tile_m` 64) it uses 220, against 56 for the
  16-wide tile. Local loads and stores are zero in every case, so nothing spills. Raw counters:
  `data/pr-validation-20261119/*/regcheck_*.csv`. With the fence the GB10 numbers are unchanged
  (`data/pr-validation-20261123-fence/gb10-spark5/regcheck.csv`).
- The fence uses the same gate and the same two instructions as #9759, so if #9759 merges first with a shared helper
  the change here is a one-line swap. If you'd rather keep this PR to the tile and grid only, I can split the fence
  into its own PR.
- The tile choice is derived from `output_columns`, which is already in the in-process cache key, so `_KERNEL_CACHE` is
  unchanged; only the disk compile-cache version moves.
- The order and tile could also become tuner options in `MtpFeedbackConfig`. I kept this PR to a fixed default because
  the 128-wide tile wins on both GPUs I measured from 256 tokens up.
- `_cute_prefill.py` is byte-identical in the `local-inference-lab/flashinfer` fork at `9873d7a`, which eugr's
  `spark-vllm-b12x` images build, so this patch also applies there unchanged.

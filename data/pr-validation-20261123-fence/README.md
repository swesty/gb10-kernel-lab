# B1 proxy fence: SASS check and cost (2026-11-23)

These files back the "Proxy fence" section of [`upstream/pr-b1-mtp-prefill-tile.md`](../../upstream/pr-b1-mtp-prefill-tile.md). Everything was run on spark-5 (GB10, sm_121a) with b12x 1.5.0, cutlass-dsl 4.7.0 and `cutlass.CUDA_VERSION` 13.3.

## `gb10-spark5/`: the PR file with and without the fence

Two variants were run in fresh containers from image `06e0b04bcba1`, each with its own compile cache:

- `patched`: the B1 PR file as of 2026-11-19;
- `fenced`: the same file plus the gated fence, made by `make_fenced.py`.

`fencecheck.sh` is the driver.

- `mtp.txt`: harness timings at 16 to 4,096 tokens, two alternating rounds. Each value is the median of 20 launches, each after a 96 MB L2 flush.
- `regcheck.csv`: Nsight Compute registers, shared memory and local loads/stores at 16 and 4,096 tokens.
- `pytest_fenced.txt`: `test_mtp_feedback_plus.py`, which is the upstream test file plus the new test, run on the fenced build. Result: 12 passed, 8 failed (the same 8 as without the change), 1 skipped.
- `sass_check.txt` and `sass_check_patched.txt`: output of `sass_fenced.py` for every MTP GEMM kernel in each cache. For each consumer release, it checks that `MEMBAR` and then `FENCE.VIEW.ASYNC.S` come after the last shared-memory load of that k-block.

## `fence-ab-container/`: the first A/B, inside the serving container

`make_fence_variant.py` builds a module copy in which `B12X_MTP_FENCE` selects the variant:

- `0`: no fence;
- `1`: the fallback fence for CUDA < 13.4;
- `2`: the ≥ 13.4 inline PTX, which fails to compile on 13.3 with "NVVM backend compilation failed".

The other files:

- `timing.sh`: three alternating runs per configuration, in both the PR configuration (tile 128, 2 stages, N raster) and stock (tile 16, 3 stages).
- `timing.out`: the raw results, including a 500-launch bitwise stress test per mode (0 of 500 differed in each mode).
- `ungated_fence_13_3.txt` and `make_ungated.py`: the fenced PR file with the CUDA 13.4 gate removed (the release fence always emitted), compiled in a fresh container. It fails with "NVVM backend compilation failed" on `cutlass.CUDA_VERSION` 13.3, which is why the gate is needed.
- `sass_check_fallback.txt`: release-window output for the mode-1 kernel. It looks back only 4 instructions, so it misses the fence; `fence_order.py` is the corrected check.

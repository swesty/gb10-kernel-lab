<!-- Draft issue(s) for flashinfer-ai/flashinfer (experimental/b12x). Not filed. Item 3 should be re-checked first. -->
# Smaller issues found while profiling Qwen3.8-Flash-Next on a DGX Spark

Environment: b12x 1.5.0 (`a321f9a6`), vLLM `local-inference-lab/vllm` `ab86b707`, GB10 (sm_121), CUDA 13.0. I confirmed
item 1 is still present in FlashInfer `main` @ `d3f59f0`. Items 2 to 4 were seen on 1.5.0 and I have not re-run them
against FlashInfer; each one says so below. Item 1 is also present in the `local-inference-lab/flashinfer` fork at
`9873d7a`, which eugr's `spark-vllm-b12x` images build, and the item 2 `ready_queue` code path is there too (not re-run).

## 1. `NCCL_IB_GID_INDEX=-1` is taken literally (`comm/roce/roce_oneshot.py:88`)

`default_gid_index()` reads `B12X_ROCE_GID_INDEX`, then `NCCL_IB_GID_INDEX`, then falls back to 3. NCCL uses `-1` to mean
"choose automatically", so a cluster that sets it that way ends up with GID index −1 in b12x. Treating negative values as
unset (falling through to the default or to auto-discovery) would fix it.

## 2. `B12X_DYNAMIC_WORK_SOURCE=ready_queue` does not compile

The MoE dynamic kernel fails at engine start with `PHASE_DYNAMIC_TO_STATIC_BOOL` ("Cannot use a Runtime value … where a
Python value boolean is required"). I reproduced it on 1.5.0. The later changes to `moe/_shared/kernels/dynamic.py` do not
appear to touch that path, but I have not re-run it. For comparison, `persistent_grid` compiles, and was 1.2% slower than
the default for prefill.

## 3. `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` cannot prepare the M = 1 decode plan

vLLM fails at startup in this mode on 1.5.0. In a standalone harness, the deterministic MoE was also 1.7–2.5× slower at
4,096 tokens. FlashInfer has recent deterministic-output fixes (`ecc4b8a9`, 2026-11-05), so this may already be fixed; I
have not re-run it.

## 4. The preparation cache identity uses the resolved model snapshot path

`~/.cache/b12x/compile/preparation/<hash>.json` stores the snapshot directory in `identity.namespace.model`. When the
Hugging Face ref moved to a new revision of the same model (byte-identical weight shards; only `config.json` metadata
changed), the next start found no cached selections and began a full cold autotune of 209 plans. Host memory
fell to about 9 GiB on a 121 GiB unified-memory Spark before I stopped it. Keying the identity on the model's content (a
digest of the config and the weight index) instead of its path would avoid this. At minimum, a log line such as "no cached
selections for this model identity, autotuning N plans" would make it obvious what is happening.

References: §13.2, §13.11 and §13.20 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

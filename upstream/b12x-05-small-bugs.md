<!-- Draft issue(s) for flashinfer-ai/flashinfer (experimental/b12x) and local-inference-lab/b12x. Not filed. -->
# Smaller issues found while profiling Qwen3.8-Flash-Next on a DGX Spark

Environment: b12x 1.5.0 (`a321f9a6`), vLLM `local-inference-lab/vllm` `ab86b707`, GB10 / sm_121, CUDA 13.0. Item 1 is present in FlashInfer `main` @ `d3f59f0`
and local-inference-lab/b12x `master` @ `2cc7f66`; items 2–4 were seen on 1.5.0 and are noted per item. FlashInfer has recent
deterministic-output fixes (`ecc4b8a9`), so item 3 should be re-checked there before filing.

## 1. `NCCL_IB_GID_INDEX=-1` is taken literally (`comm/roce/roce_oneshot.py:88`)

`default_gid_index()` reads `B12X_ROCE_GID_INDEX`, else `NCCL_IB_GID_INDEX`, else 3. NCCL uses `-1` to mean "choose
automatically", and fleets that set it that way end up with GID index −1 in b12x. Suggest treating negative values as
"unset" (fall through to the default or to auto-discovery).

## 2. `B12X_DYNAMIC_WORK_SOURCE=ready_queue` does not compile

The MoE dynamic kernel fails at engine start with `PHASE_DYNAMIC_TO_STATIC_BOOL` ("Cannot use a Runtime value … where a
Python value boolean is required"). Reproduced on 1.5.0; master's `moe/_shared/kernels/dynamic.py` changes do not appear
to touch that path, but we have not re-run it. (`persistent_grid` compiles; it was 1.2% slower than the default for
prefill.)

## 3. `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` cannot prepare the M = 1 decode plan

vLLM fails at startup in that mode. In a standalone harness the deterministic MoE was 1.7–2.5× slower at 4,096 tokens.
Not re-run on master.

## 4. Preparation cache identity uses the resolved model snapshot path

`~/.cache/b12x/compile/preparation/<hash>.json` → `identity.namespace.model` holds the snapshot directory. When the Hugging
Face ref moved to a new revision of the same model (byte-identical weight shards, only `config.json` metadata
changed), the next start found no cached selections and began a full cold autotune of 209 GDN-prefill plans while host
memory fell to ~9 GiB on a 121 GiB unified-memory Spark. Keying the identity on model content (config + weight index
digest) instead of the path would avoid that. At minimum, a clear log line ("no cached selections for this model
identity, autotuning N plans") would help.

References: §13.2, §13.11, §13.20 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

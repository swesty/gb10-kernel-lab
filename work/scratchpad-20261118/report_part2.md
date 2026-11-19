
---

## 5. How vLLM drives these kernels

All citations in this section are to vLLM `main` @ `f39d833`.

### 5.1 Selection

- **Flags.** `--attention-backend X`, or `--attention-config.backend X` / `-ac.backend X`. Setting both raises "attention_backend and attention_config.backend are mutually exclusive" [SRC] `vllm/engine/arg_utils.py:1121, 1918, 2764-2774`. Names are case-insensitive [SRC] `vllm/config/attention.py:273-285`.
- **A named backend** is validated alone; failure raises `"Selected backend {x} is not valid for this configuration. Reason: {...}"` [SRC] `vllm/platforms/cuda.py:471-490`.
- **Automatic selection** walks a priority list and takes the first backend with no validation objection [SRC] `cuda.py:417-458, 520-523`.
- **Priority for capability major 12 [SRC]** `cuda.py:131-135, 166-178`:
  - Standard attention: `FLASH_ATTN`, `FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TURBOQUANT`.
  - MLA: `TRITON_MLA`, `FLASHINFER_MLA_SPARSE_SM120`.
- **`B12X` is in no list** [SRC] `cuda.py:81-178`.

Validation runs 22 checks in order: head size, model dtype, KV dtype, block size, multimodal prefix, MLA, sinks, sparse, per-head scales, compute capability, attention type, sliding window, and so on, ending with a backend-specific `supports_combination` [SRC] `vllm/v1/attention/backend.py:296-389`.

### 5.2 `FLASH_ATTN` on capability 12

- **Version.** FA3 if major 9; FA4 if major 10; otherwise FA2 [SRC] `vllm/v1/attention/backends/fa_utils.py:91-100`.
- **What FA2 cannot serve, and the reason string [SRC]:**
  - **Sinks.** `supports_sink()` is true only for FA3 or FA4 (`flash_attn.py:422-426`; `fa_utils.py:244-247`).
  - **FP8 KV.** "FP8 KV requires FA3 (SM90), equal-d512 FA4 (SM90), or FA4 (SM100)." (`flash_attn.py:464-474`).
  - **Multimodal prefix.** Requires FA4 (`flash_attn.py:418-420`).
- **What it serves:** sliding window, non-causal attention, batch invariance, decode context parallelism, head sizes up to 256 that are multiples of 8, block sizes that are multiples of 16 [SRC] `flash_attn.py:332-406, 1038`.
- **CUDA graphs.** FA2 reports `UNIFORM_BATCH`, not `ALWAYS`, because FA2 has special packed-GQA handling for single-token queries [SRC] `flash_attn.py:578-597`.

**Consequence [INF].** On SM12x, automatic selection yields FA2 for a plain BF16 model and `FLASHINFER` as soon as the model needs FP8 or NVFP4 KV, sinks, or a multimodal prefix.

**Forcing FA4.** The gate is in vLLM's tree: "FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x" [SRC] `vllm/vllm_flash_attn/flash_attn_interface.py:72-86`. With `--attention-config.flash_attn_version=4` on capability 12, `get_flash_attn_version` logs "FlashAttention 4 is not supported (...); it will not be used.", fails an assertion that is caught, and returns `None` [SRC] `fa_utils.py:189-199`. Validation does not require a version, so the backend is still selected, and the first forward pass fails on `"FlashAttention version not detected."` [INF from SRC] `flash_attn.py:1201-1203`. Expected outcome: a startup failure, not a silent fallback. Not run.

### 5.3 `FLASHINFER` on SM12x

**Capability window.** `>= (8, 0)` and `<= (12, 1)` [SRC] `vllm/v1/attention/backends/flashinfer.py:779-781`.

**Prefill.** `supports_trtllm_attention(is_prefill=True)` is false on the SM12x family, so prefill always uses `BatchPrefillWithPagedKVCacheWrapper(..., backend="auto")` [SRC] `vllm/utils/flashinfer.py:884-887`; `flashinfer.py:1573-1581`. In FlashInfer, `"auto"` resolves to `"fa2"` for any capability major other than 9 [SRC] `flashinfer/utils.py:717-730`. vLLM's own comments call SM12x prefill "fa2-only" [SRC] `flashinfer.py:1276-1277`.

**Decode uses XQA only if all of these hold [SRC]:**

1. `--attention-config.use_trtllm_attention` is not false.
2. `VLLM_BATCH_INVARIANT` is unset.
3. **`has_nvidia_artifactory()` is true.** That is: the `flashinfer_cubin` package is importable, or `VLLM_HAS_FLASHINFER_CUBIN=1`, or an HTTP request to the cubin repository returns 200 within 5 seconds (`vllm/utils/flashinfer.py:61-64, 85-92, 840-864`).
4. Query heads are a multiple of KV heads.
5. Head dimension is 16 to 256 in steps of 16 (`flashinfer.py:840-841`).
6. Decode context parallelism is off (`flashinfer.py:1122-1132`).

Otherwise decode uses `BatchDecodeWithPagedKVCacheWrapper` with tensor cores and `backend="auto"` [SRC] `flashinfer.py:1626-1645`.

**Consequence [INF].** On an air-gapped Spark without the cubin package, XQA is silently off. Sinks then become unsupported, because sink support on SM12x is defined as XQA being available [SRC] `flashinfer.py:783-796`, and CUDA-graph support drops from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE` [SRC] `flashinfer.py:1309-1352`.

**Query dtype on SM12x [INF from SRC]** `flashinfer.py:1264-1305`: the model dtype in every case. With an FP8 cache, "Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries". With XQA, "requires BF16/FP16-Q even with FP8 KV cache".

**Other SM12x specifics [SRC]:**

- Block sizes 16, 32, 64. Sizes of 128 and above are advertised only on the SM100 family (`flashinfer.py:595-612`).
- NVFP4 KV: only the literal `nvfp4` dtype; head size 64 and sinks are refused; decode context parallelism and non-causal attention raise (`flashinfer.py:727-759, 1048-1051, 1517-1521`).
- Multimodal prefix: served natively by `BatchPrefillWithCausalBidirectionalRangesWrapper` (`flashinfer.py:1585-1603`).
- Speculative verification: ragged multi-token queries stay on the XQA decode path with a packed draft mask (`flashinfer.py:1465-1506, 1731-1743`).
- Batch invariance: `FLASHINFER` fails validation (`backend.py:221-223`).

### 5.4 Multimodal-prefix models on capability 12

Both composite backends are invalid: `TRITON_FLASH_ATTN` requires major 9 and `TRITON_FLASHINFER` requires major 10 [SRC] `triton_flash_attn.py:55-61`; `triton_flashinfer.py:12-22`. The resulting chain [INF from SRC]:

1. `FLASH_ATTN` fails (needs FA4).
2. `FLASHINFER` serves it, unless the model uses soft-capped logits, an unsupported head size, or sinks.
3. Otherwise `TRITON_ATTN`.

### 5.5 Log lines to grep [SRC]

| Situation | Format string | Location |
|---|---|---|
| Named backend | `"Using %s backend."` | `cuda.py:489` |
| Automatic | `"Using %s attention backend out of potential backends: %s."` | `cuda.py:549-557` |
| FlashAttention version | `"Using FlashAttention version %s"` | `flash_attn.py:1083-1086` |
| FlashInfer kernels | `"FlashInfer resolved query dtypes: prefill=%s, decode=%s, decode_backend=%s, kv_cache_dtype=%s, arch=%s"` | `flashinfer.py:1216-1224` |
| b12x | `"Using b12x with q_heads=%d kv_heads=%d head_dim_qk=%d ... planned_page_sizes=%s ... scratch=%d bytes."` | `b12x.py:718-731` |

In the FlashInfer line, `decode_backend` is `xqa`, `trtllm-gen` or `flashinfer-native`, and `arch` prints `sm120` or `sm121`.

---

## 6. How SGLang drives these kernels

All citations in this section are to SGLang `main` @ `b2cc023f`.

### 6.1 Defaults

- **Standard attention on SM12x:** `flashinfer`, or `triton` if the model has attention sinks. The source comment: "FlashInfer does not support attention sinks." [SRC] `python/sglang/srt/arg_groups/model_override_base.py:367-371`.
- **MLA on SM12x:** `triton` [SRC] `model_override_base.py:372-388`.
- For comparison: SM90 gets `fa3`; SM100 gets `trtllm_mha`, or `fa4` for asymmetric KV.

### 6.2 The `flashinfer` backend

Prefill and decode both use `backend="fa2"` on the paged wrappers. Ragged prefill uses `"auto"`. The only architecture branch is for SM100 [SRC] `python/sglang/srt/layers/attention/flashinfer_backend.py:373-374, 550-616`.

### 6.3 `trtllm_mha`

- **Kernels.** FMHAv2 for prefill, XQA for decode, both gated on `get_platform().is_sm90 or get_platform().is_sm120` [SRC] `python/sglang/srt/layers/attention/trtllm_mha_backend.py:6-12, 309, 417`.
- **Prefill restriction,** verbatim: "TRTLLM FMHAv2 prefill on SM120 does not support fp8_e4m3 KV cache or skip-softmax." The check is on prefill only; XQA decode with FP8 KV is not rejected [SRC] `python/sglang/srt/arg_groups/attention_hook.py:216-230`.
- **Page size.** 16, 32, 64 or 128; anything else snaps to 64. The default of 1 is therefore changed to 64 with a warning [SRC] `python/sglang/srt/arg_groups/overrides.py:1387-1400`.
- **Speculative decoding.** Top-k must be 1 [SRC] `python/sglang/srt/arg_groups/speculative_hook.py:1369-1373`.
- **Prefill copies pages.** FMHAv2 derives strides from shapes, so SGLang first copies the batch's pages into a `[pages, 2, page_size, heads, head_dim]` tensor [SRC] `trtllm_mha_backend.py:1565-1582`.

### 6.4 Per-phase flags

`--prefill-attention-backend` and `--decode-attention-backend` override `--attention-backend` for their phase. Differing phases get a hybrid wrapper [SRC] `python/sglang/srt/arg_groups/fields/exec_.py:237-275`; `model_override_base.py:154-168`.

Combinations rejected on SM12x [SRC]:

| Combination | Result | Citation |
|---|---|---|
| `fa3`, either phase | AssertionError | `attention_registry.py:311-318` |
| `trtllm_mha` prefill + FP8 KV or skip-softmax | ValueError | `attention_hook.py:216-230` |
| `trtllm_mha` + speculative top-k > 1 | ValueError | `speculative_hook.py:1369-1373` |
| `fa4` + FP8 KV | AssertionError at first call | `flash_attn/cute/interface.py:787-790` |
| `fa4` + top-k > 1 + page size > 1 | ValueError | `speculative_hook.py:1464-1475` |
| `--kv-cache-dtype nvfp4` + explicit prefill other than `flashinfer` | ValueError | `kv_cache_hook.py:61-79` |
| `--kv-cache-dtype nvfp4` + explicit decode other than `trtllm_mha` | ValueError | `kv_cache_hook.py:81-87` |
| Hybrid GDN model + `fa4` | AssertionError | `attention_registry.py:536-550` |
| Deterministic inference + `trtllm_mha` | ValueError | `overrides.py:1288-1306` |

`fa4` prefill with `trtllm_mha` decode passes every argument-level check for a plain model with a BF16 cache. Run-time compatibility of the shared cache layout was not verified [INF].

### 6.5 Two helpers with the same name and different meanings

| Call | True on SM120 | True on SM121 | Citation |
|---|---|---|---|
| `get_platform().is_sm120` (maps to `is_sm120_supported`) | yes | **yes** | `runtime_context.py:1855`; `utils/device.py:311-315` |
| `device.is_sm120()` | yes | **no** | `utils/device.py:355-358` |
| `device.is_sm121()` | no | yes | `utils/device.py:361-365` |
| `is_blackwell_supported()` | yes | yes | `utils/device.py:301-308` |

Section 10 lists the call sites where this changes behaviour on a Spark.

---

## 7. MLA and sparse attention

MLA (multi-head latent attention) stores a compressed latent per token. Sparse variants attend only to a top-k subset of the context chosen by an indexer. Neither uses the kernels in section 4.

### 7.1 vLLM

| Backend | Gate | KV dtypes | Block sizes | FlashInfer call | Citation |
|---|---|---|---|---|---|
| `TRITON_MLA` | any | `auto`, FP16, BF16, FP8 | multiple of 16 | none | `mla/triton_mla.py:124-165` |
| `FLASHINFER_MLA_SPARSE_SM120` | major 12 | `auto`, `fp8`, `fp8_e4m3`, `fp8_ds_mla` | multiple of 64 | `trtllm_batch_decode_with_kv_cache_mla` | `mla/flashinfer_mla_sparse.py:195-225` |
| `FLASHINFER_MLA_SPARSE_DSV4` | major 10 or 12 | on major 12: FP8 variants only | 256 | `trtllm_batch_decode_sparse_mla_dsv4` | `models/deepseek_v4/nvidia/flashinfer_sparse.py:118-166` |
| `FLASHINFER_MLA_SPARSE_DSV41` | major 10 or 12 | on major 12: FP8 variants only | 64 on SM12x, else 128 | same | `models/deepseek_v41/nvidia/flashinfer_sparse.py:66-116` |

Details [SRC]:

- **Dense MLA resolves to `TRITON_MLA`; sparse MLA to `FLASHINFER_MLA_SPARSE_SM120`** [INF from the sparse-flag check].
- **The SM120 sparse backend always stores an FP8-packed cache.** `auto`, `fp8` and `fp8_e4m3` are canonicalized to `fp8_ds_mla` (`model_executor/layers/attention/mla_attention.py:383-388`).
- **It requires `index_topk == 2048`** (`mla/flashinfer_mla_sparse.py:239-272`) and 64-row pages: "The SM120 kernels are built for 64-row pages only." (`mla/flashinfer_mla_sparse_sm120.py:210-211`).
- **DeepSeek V4 and V4.1 are model-driven.** With no explicit backend, major 12 returns the FlashInfer SM120 class (`models/deepseek_v4/nvidia/model.py:556-579`).
- **MLA prefill off SM100 is FlashAttention only**, which on SM12x means FA2 with V zero-padded to the QK head dimension (`platforms/interface.py:472-492`; `mla/prefill/flash_attn.py:537-546`).
- **`backend="cake"` and `kv_cache_format="nvfp4"` are not used on vLLM's SM12x MLA paths.** The NVFP4 MLA cache dtype belongs to a backend gated on major 10 (`mla/flashinfer_mla_sparse.py:99-122`). FlashInfer ships these SM12x kernels; vLLM `main` does not call them.

### 7.2 SGLang

- **`trtllm_mla`**: the argument layer admits SM12x ("only supported on Blackwell GPUs (SM100/SM12x)") [SRC] `overrides.py:1417-1431`. Issue #40286 reports the underlying call failing on SM121 with `TllmGenFmhaRunner ... Unsupported architecture` [ISSUE]. Dense `trtllm_mla` likely fails the same way [INF]; not verified.
- **DeepSeek V4**: the model override forces `attention_backend = "dsv4"` and page size 256. On SM12x the sparse call goes to `flash_mla_with_kvcache_sm120` [SRC] `deepseek_v4_backend.py:4578-4582`.
- **`SGLANG_SM120_FLASHMLA_BACKEND`**: `"flashinfer"` (default), `"triton"`, or anything else, which falls through to a pure-PyTorch path. The variable is not validated, so a typo silently selects PyTorch [SRC] `environ.py:1447-1448`; `flash_mla_sm120.py:298-374`.
- **GLM sparse attention with FP8 KV** selects `flashinfer_sparse_mla`, valid only for two `GlmMoeDsa*` model classes [SRC] `overrides.py:688-722`; `flash_mla_sm120.py:29-32`.

### 7.3 FlashInfer's SM12x sparse kernels

| Kernel | MMA | Staging | SM121 | Citation |
|---|---|---|---|---|
| Cake DeepSeek-V4 NVFP4 sparse MLA | BF16 k16, FP8 k32, `mxf4nvf4` k64 | `cp.async` ring; TMA-free | README says SM120 / SM121 | `csrc/cake_dsv4/README.md:107-211` |
| Hand-written sparse MLA SM120 | same three | 1-D `cp.async.bulk` plus `cp.async`; split grid and LSE merge | not determined | `include/flashinfer/attention/sparse_mla_sm120/` |
| XQA MLA SM120 | not determined | TMA, page id as the last coordinate; 384 threads | not determined | `csrc/xqa/mla_sm120.cu:132` |
| `cake_vsa_sm120` (block-sparse) | `m16n8k16` | TMA 4-D with `fence.proxy.async` | yes, `sm_120a` and `sm_121a` | `flashinfer/jit/cake_vsa_sm120.py:15, 42` |
| SM120 DSA indexer | FP8 k32; `mxf4nvf4` k64 | TMA 2-D/3-D; `kSmemLimit = 101376` | "on SM12x GPUs" | `flashinfer/experimental/sm120_dsa_indexer/` |
| DeepGEMM SM120 paged MQA logits | FP8 k32 | TMA; 384-thread persistent program | README: SM120a only | `flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md:10-20` |
| `nvfp4_attention_sm120` (dense, non-causal) | `mxf4nvf4` k64 | TMA header present | yes | `flashinfer/nvfp4_attention_sm120.py:171` |

The DeepGEMM README states the constraint in one line: "no tcgen05/TMEM/WGMMA, which `ptxas` rejects on this architecture".

**Experimental policy.** `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` gates only *automatic* selection of experimental backends; an explicit `backend="<name>"` always works [SRC] `flashinfer/api_logging.py:2600`; `flashinfer/experimental/__init__.py:26-30`. Experimental features are "JIT-only and must not ship in pre-built packages" [SRC] `flashinfer/experimental/AGENTS.md:24`. The variable does not appear in vLLM's tree.

---

## 8. Open work, pull request by pull request

States were read from the GitHub API on 2026-11-17.

### 8.1 Upstream FlashAttention: two competing lines

**Line A extends the Ampere subclass.**

| PR | State | Size | What it adds | Notes |
|---|---|---|---|---|
| #2329 | merged 2026-03-12 | +102/−7 | The SM120 subclass; fixes an SM80 causal bug that reset the running maximum | Correctness only, on GB10: 17/17 configurations |
| #2336 | open, last update 2026-08-14 | +262/−178 | Split-KV, writing FP32 partials from accumulator registers | Maintainer review: "The paged KV stuff doesnt' look right, we should not be indexing in pytorch... For split we should be writing in fp32, not bf16." Author reports both addressed |
| #2348 | open, last update 2026-11-01 | +561/−224 | In-kernel paged KV; moved to 2 stages | A commenter first measured 10× dense; maintainer: "In the past we've see 0-10% perf diff between dense and paged KV" |
| #2980 | open | +1885/−641 | FP8 e4m3 `m16n8k32`; split-KV; FP8-only paged KV | BF16 paged calls rejected explicitly |
| #2998 | open | +17/−0 | Tile selection: falls back to 64×64, fixing head dim 256 | No comments |
| #3060 | open | +310/−40 | Paged KV for SM80 and SM120 | Reviewer found two correctness bugs on an RTX 5070 Ti; author acknowledged |

**Line B replaces the subclass with SM12x-owned kernels.**

| PR | State | Size | What it adds |
|---|---|---|---|
| #3108 | open | +342/−18 | Groundwork: SM count from the stream context, per-batch split-KV pieces |
| #3109 | open | +16,216/−290 | New kernels under `flash_attn/cute/sm120/`: warp-MMA forward **with TMA K/V**, head dims to 256, paged KV, block sparsity, split-KV, a transposed paged-decode kernel, and a new backward. The author states the forward is a port of the SGLang SM12x implementation |
| #3132 | open, draft | +17,412/−320 | FP8 (e4m3 and e5m2) forward for the #3109 kernels |

Line B's FP8 pull request states it is not stacked on #2980 and that #3109 replaces the file #2980 edits. The two lines conflict.

**All open pull requests show `REVIEW_REQUIRED`.** The only maintainer comments found are on #2336 and #2348.

A separate FA2 pull request, #3026 (+3/−2, open), adds capability 12.0 to FA2's `is_sm8x` predicate so SM120 gets the Ada tiles. Its rationale: "64KB per block means only 1 CTA per SM on `sm_120`, 4 of 48 warps". **It explicitly does not cover capability 12.1** [PR].

### 8.2 SGLang

- **#32991** (merged 2026-08-13, +10,449/−198): the `fa4_sm120` kernels of section 4.7. A later comment reports a bug fixed in #49044, which was not read.
- **#45871** (open, +649/−68, zero comments, CI red at reading): native FP8 KV for those kernels. QK and PV on FP8 `m16n8k32`; V kept as FP8 in shared memory; P pre-scaled by 256 before the FP8 cast; new tile candidates for head dimensions above 256.
- **#45262** (merged 2026-11-10): b12x for MXFP8 linear layers on SM12x. GEMM, not attention.
- **#35899** (closed, not merged): a b12x compressed-MLA attention branch for DeepSeek-V4-Flash.

### 8.3 FlashInfer

- **#7675** (open, filed 2026-10-28). In the b12x MXFP8 **GEMM** on SM120, "the TMA refill of a pipeline slot is not ordered after the MMA warps' ldmatrix". Under CUDA-graph replay with concurrent work on a second stream, wrong tiles are returned; eager results match CUTLASS. The reporter's sweep: 1,537 bad replays stock, 0 with a one-line `fence_proxy("async.shared", space="cta")`. Reproduced on an RTX 5090 [ISSUE] [S18].
- **#7676** (open): the fence fix [S19].
- **#6977** (open): b12x paged attention cannot plan head dimension 64 decode [S20].
- **#3170** (open, updated 2026-11-13): the DGX Spark support audit, the canonical gap list [S17].

**Does the #7675 hazard apply to b12x paged attention? [INF, unresolved]**

- *Observation:* the paged attention files contain no `fence_proxy` call.
- *Mechanism that would cause the bug:* a math warp reads a slot with `ldmatrix` (generic proxy), the stage is released, and the next TMA write (async proxy) to the same slot is issued without a fence.
- *Why it might not apply:* in the default decode kernel the same warp that reads also issues the next TMA, after releasing the stage, and the pipeline has one stage. In the GEMM, the producer and consumers are different warps.
- *Why it might:* the role-specialized decode variant has a separate loader warp, which is the GEMM's structure.
- *Discriminating test:* the #7675 reproduction pattern (CUDA-graph replay with a copy and matmul on a second stream) applied to b12x paged decode, with and without the role-specialized variant.

### 8.4 SGLang issues on these GPUs

- **#40286** (open): GLM-5.3-Flash on 2× DGX Spark. Every sparse-attention backend fails: the automatic one with `Unsupported architecture`; TileLang with `Failed to set the allowed dynamic shared memory size to 169984`; the FlashInfer sparse backend because the model class is not in an allow-list. The allow-list is unchanged at the commit read [ISSUE]; [SRC] `flash_mla_sm120.py:29-32`.
- **#42364** (open): DeepSeek-V4.1 on 4× DGX Spark fails at attention initialization because a `major >= 10` test sent SM12x down a path requiring an SM100-only DeepGEMM feature. The commit read contains the fix shape (`if not is_sm100() ...: return dense_blocks`), so the failure appears fixed on `main` although the issue is open [INF from SRC] `dsv4/v41_indexer/__init__.py:139-145`.
- **#50749** (open, one day old): a TileLang kernel on an RTX 5090 requests 170,048 B of shared memory against 101,376 B. The two KV buffers alone need 2 × 64 × (512 + 64) × 2 = 147,456 B [ISSUE].

---

## 9. Published performance numbers

**Every number here is an author's claim from a pull request or issue.** None was reproduced. Hardware is as the author named it.

### 9.1 BF16 kernels against FA2 and against each other

| Source | Hardware | Comparison | Result |
|---|---|---|---|
| FA #3109 | RTX PRO 6000 (188 SMs) | new SM12x kernels vs upstream FA4 main, forward, 111 cases | 1.34× geomean |
| FA #3109 | RTX PRO 6000 | vs FA2 2.8.4, forward | 1.13× |
| FA #3109 | RTX PRO 6000 | decode on a dense cache vs upstream FA4 main | 2.54× (up to 12.3×) |
| FA #3109 | RTX PRO 6000 | decode vs FA2 dense / FA2 paged | 1.15× / 1.12× |
| FA #3109 | GB10 (48 SMs) | head dim 128 GQA, 8,192 tokens, causal forward vs main | 1.16× |
| FA #3109 | GB10 | decode, one 32K-context request vs main | 1.7–2.0× |
| FA #2998 | RTX PRO 6000 | tile tuning vs default FA4, 49 shapes | 1.19× (range 0.997–1.87×) |
| FA #2998 | RTX PRO 6000 | tile tuning vs FA2 | 1.11× |
| FA #3026 | RTX PRO 6000 Server Edition | FA2 with Ada tiles vs FA2 as shipped | 1.02–1.09× non-causal; ≈1.00× causal |
| FA #2980 | RTX PRO 6000 | BF16 decode vs FA2, 108 shapes | 1.17× (61/108 faster; worst 0.43×) |
| FA #2348 | GB10 | paged vs dense decode, page size 64 | 1.09–1.18× slower |
| SGLang #32991 | RTX PRO 6000 | `fa4_sm120` prefill vs Triton, 228 cases | 3.03× |
| SGLang #32991 | GB10 | `fa4_sm120` prefill vs Triton | 3.61× |
| SGLang #32991 | RTX PRO 6000 / GB10 | prefill vs the old SM80-derived path | 2.38× / 1.73× |
| SGLang #32991 | RTX PRO 6000 / GB10 | paged decode vs Triton, page size 64 | 1.31× / 1.21× |

One absolute data point from SGLang #32991, head dimension 256, 6,144 tokens: on GB10 the new kernel took 2,108 µs against 1,824 µs for the old fallback, so the new kernel **lost** in that case. On the RTX PRO 6000 the same case was 456 µs against 572 µs.

### 9.2 FP8 against BF16

| Source | Hardware | Comparison | Result |
|---|---|---|---|
| FA #2980 | RTX PRO 6000 | FP8 prefill vs FA2 BF16, 49 rows | 1.31× |
| FA #2980 | RTX PRO 6000 | FP8 decode vs FA2, 108 shapes | 1.70× (103/108 faster; worst 0.85×) |
| FA #2980 | RTX PRO 6000 | FP8 paged vs FP8 contiguous | 0.883× |
| FA #3132 | RTX PRO 6000 | FP8 vs BF16 prefill, 40 cases | 1.69× |
| FA #3132 | RTX PRO 6000 | FP8 vs BF16 paged decode, page 256 | 1.84× |
| SGLang #45871 | RTX PRO 6000 | FP8 decode, head dim 256, 9,216 KV tokens | 2.03× |
| SGLang #45871 | RTX PRO 6000 | FP8 prefill, head dim 512 | 2.19× |
| SGLang #45871 | RTX PRO 6000 | serving `gemma-4-31B-it`, 8k in / 1k out | 2.25× throughput (383 → 864 tok/s) |

SGLang #45871 also reports GSM8K-Platinum accuracy of 0.730 for BF16 and 0.742 for FP8.

### 9.3 What the numbers suggest [INF]

- **Against FA2, BF16 gains on the RTX PRO 6000 are 1.1× to 1.2×.** FA2 is not far behind on this card, despite running inherited tiles.
- **The largest BF16 gains are against weak baselines:** Triton, or upstream FA4's untuned single-split SM120 path.
- **FP8 is the consistent large gain,** 1.7× to 2.0× across three independent pull requests.
- **There are no FP8 numbers for the DGX Spark.** Every FP8 result is from an RTX PRO 6000.
- **GB10 gains are smaller than RTX PRO 6000 gains for prefill** (1.16× against 1.34× in #3109; 1.73× against 2.38× in SGLang #32991), with one outright loss.

---

## 10. Where SM120 and SM121 diverge

Most code treats the two machines alike. These are the places it does not.

### 10.1 Tuned for 12.0, generic on 12.1

| Where | What happens on a Spark | Citation |
|---|---|---|
| b12x paged trait specializations (5 of them) and one planner budget | generic tile sweep instead of the tuned tiles | `flashinfer/experimental/b12x/attention/paged/traits.py:383, 411, 459, 468, 486`; `planner.py:272` |
| vLLM Triton attention, 64 parallel softmax segments | not applied | `vllm/v1/attention/backends/triton_attn.py:152-162` |
| FA2 tile fix (FA #3026) | explicitly not covered | [PR] |
| SGLang split-KV latency model | constants fitted on 110-SM and 188-SM discrete cards | `sglang/.../fa4_sm120/runtime.py:45-48` |

### 10.2 Excluded on 12.1 in SGLang (`device.is_sm120()` is false)

| Where | Effect | Citation |
|---|---|---|
| Qwen sparse attention decode via the TRT-LLM kernel | disabled; comment: "it silently corrupts long-context decode on SM121/GB10" | `srt/layers/attention/qwen_sparse_attn_backend.py:108` |
| FP4 MoE fused-finalize fast path | SM120 only | `srt/layers/moe/moe_runner/flashinfer_cutlass.py:284, 323` |
| SM120 BF16 GEMM, FP8 and NVFP4 GEMM kernels | not used | `kernels/ops/gemm/sm120_bf16.py:59-60`; `kernels/kda_kernels/sm120_fp8.py:129` |
| Dense block-FP8 GEMM routing | SM121 keeps the DeepGEMM route | `srt/model_loader/utils.py:285` |

### 10.3 Specific to 12.1

| Where | Effect | Citation |
|---|---|---|
| SGLang Qwen sparse variable-length attention | uses a dedicated `qwen38_qsa_sm121_varlen` kernel | `qwen_sparse_attn_backend.py:121-124` |
| SGLang MoE kernel | 3 stages instead of 4: "sm_121 ... caps shared memory at 99 KB per block and num_stages=4 needs 108 KB" | `kernels/ops/moe/inkling_moe.py:973` |
| SGLang FA4 split heuristic | integrated-GPU branches; "Three unsplit CTAs per LPDDR channel already saturate the LPDDR path" | `fa4_sm120/runtime.py:125-146, 245-248` |
| vLLM default batch tier | GB10 kept on the small `max_num_batched_tokens` tier | `vllm/engine/arg_utils.py:3004-3006` |
| FlashInfer wheels | CUDA-13 aarch64 wheels carry no native `sm_121a` binaries; the Spark runs SM120-family binaries | [VAULT, from v0.6.18 release notes] |
| FlashInfer b12x FP4 **GEMM** auto-selection | SM121 excluded by policy; comment says cutlass and cuDNN are faster on GB10 in most cases | [VAULT, source inspection 2026-09-10] |

### 10.4 What the divergence means [INF]

The Spark gets correct kernels almost everywhere. It gets tuned kernels almost nowhere. The single strongest piece of evidence about what limits it is SGLang's own comment that three unsplit blocks per LPDDR channel "already saturate the LPDDR path": the authors of the most Spark-aware heuristic found memory, not compute, to be the ceiling for decode.

---

## 11. Research path

### 11.1 Framing

The earlier version of this plan proposed inventing page-aligned TMA staging and SM12x split-KV. Both exist. The questions that remain open are narrower and better defined:

1. Which existing kernel wins for which workload on each machine, measured on one harness.
2. Whether the DGX Spark's limit is memory bandwidth, and what follows if so.
3. Where the unfinished pieces are: prefill staging on a paged cache, native FP8 in the paths the engines actually call, Spark-specific tuning, and dispatch.

### 11.2 The bandwidth argument for the Spark

**Arithmetic [CALC].** A hypothetical model with 32 layers, 8 KV heads and head dimension 128 stores 8 × 128 × 2 × 2 = 4,096 bytes per token per layer in BF16. That is 128 KiB per token, and 4 GiB at a 32,768-token context. One decode step reads all of it.

| Cache | Bytes read per step | Floor at 273 GB/s (Spark) | Floor at 1,792 GB/s (RTX PRO 6000) |
|---|---|---|---|
| BF16 | 4.29 GB | 15.7 ms | 2.4 ms |
| FP8 | 2.15 GB | 7.9 ms | 1.2 ms |
| NVFP4 (≈4.5 bits) | 1.21 GB | 4.4 ms | 0.7 ms |

These are floors for reading the cache once. They ignore compute.

**Hypothesis A.** On the Spark, long-context decode is bandwidth-bound for every kernel. Kernel choice matters little; cache precision matters almost linearly.

- *Supporting evidence:* SGLang's integrated-GPU comment (section 10.3); the smaller GB10 gains in section 9; the bandwidth ratio (6.6×) exceeding the SM ratio (3.9×).
- *Falsified if:* two BF16 kernels differ by more than about 20% at batch 1 and a 32K context on the Spark.

**Hypothesis B.** On the RTX PRO 6000, batch-1 decode is occupancy-bound, so split-KV and tile choice matter.

- *Supporting evidence:* #3109's 2.54× decode gain over an unsplit baseline; #2980's worst case of 0.43× at batch 32, where splitting should not help.
- *Falsified if:* achieved bandwidth is already near 1,792 GB/s without split-KV.

### 11.3 Phase 0 — Settle the device facts

Four device attributes resolve the documentation conflict in section 2.5 and the derived SM counts. On each machine:

```
cudaDevAttrMultiProcessorCount
cudaDevAttrMaxSharedMemoryPerMultiprocessor
cudaDevAttrMaxSharedMemoryPerBlockOptin
cudaDevAttrMaxBlocksPerMultiprocessor
```

- **Why first:** every residency claim in this report depends on the per-SM value. FA2's split heuristic assumes two blocks per SM. b12x and FlashInfer's fa2 read the value at run time, so their behaviour differs between a 100 KB and a 128 KB device.
- **Done when:** a four-row table exists for each machine.

### 11.4 Phase 1 — Harness

- **Record per run:** capability; engine and library commits; the log lines of section 5.5; head dimension; query-to-KV head ratio; page size; KV dtype; query length, KV length, batch.
- **Correctness oracle:** FP32 `scaled_dot_product_attention` on a de-paged copy. Per-row maximum absolute and relative error with a per-dtype tolerance. For scale: SGLang #32991 reports a maximum absolute difference of 0.0078 on RTX cards and 0.0156 on GB10 against Triton.
- **Edge cases:** lengths at page size ± 1; contexts long enough to recycle every pipeline slot; sinks under split-KV (b12x applies the sink only in the chunk owning the causal limit; XQA applies sinks in-kernel only when not split).
- **Stress gate:** every case under CUDA-graph replay with a copy and matmul on a second stream, the pattern from FlashInfer #7675.
- **Two levels:** a kernel microbenchmark and a served benchmark with time to first token and inter-token latency reported separately.

### 11.5 Phase 2 — Baseline the existing kernels

| Engine | Paths to run |
|---|---|
| vLLM | no flag; `FLASH_ATTN`; `FLASHINFER` with XQA on; `FLASHINFER` with `use_trtllm_attention=0`; `b12x`; `b12x` with `B12X_TURBO_ATTN=1` and FP8 KV; `TRITON_ATTN` |
| SGLang | no flag; `fa4` at page size 1 and 64; `trtllm_mha`; `fa4` prefill with `trtllm_mha` decode; `triton` |

- **Model:** one BF16 GQA model with head dimension 128, then one with head dimension 256.
- **Shapes:** prefill at 512, 4K and 32K tokens; decode at batch 1, 8 and 64 with 1K, 8K, 32K and 128K context; KV dtype BF16 and FP8.
- **Tools:** Nsight Compute per kernel: memory throughput, tensor-core utilization, achieved occupancy, shared memory, registers per thread.
- **Output:** a table for FlashInfer #3170. Maintainers mostly lack a GB10.

### 11.6 Phase 3 — FA2 tiles on SM12x (small, concrete)

- **Finding:** FA2 gives capability 12 the A100/H100 tile. FA #3026 fixes 12.0 only and reports 1.02–1.09×.
- **Question:** does the Ada tile (128×32 non-causal, 64×64 causal) help on a Spark, where #3026 does not apply?
- **Why it matters:** FA2 is vLLM's default. The change is three lines in upstream and must also land in vLLM's fork.
- **Dependency:** the phase-0 per-SM value decides whether 48 KB tiles double occupancy.

### 11.7 Phase 4 — Spark tuning for b12x and `fa4_sm120`

- **Finding:** b12x's specializations are keyed on (12, 0). SGLang's split model was fitted on discrete cards.
- **Experiments:**
  - Sweep b12x `num_mma_kv`, `graph_ctas_per_sm` and chunk budgets on 48 SMs.
  - Sweep `fa4_sm120` split counts on GB10 and compare against the fitted model's choice.
  - Test whether the "three unsplit CTAs per LPDDR channel" rule holds across head dimensions.
- **Contribution:** a (12, 1) specialization table for b12x, or evidence that the generic sweep is already near-optimal.

### 11.8 Phase 5 — Prefill staging on a paged cache

- **Finding:** decode has page-aligned TMA in b12x and in one SGLang specialization. Prefill does not. b12x extend uses per-thread `cp.async` with a per-row lookup; its own comment names the missing piece ("until extend TMA gets the decode kernel's page-entry flattening"). SGLang uses a `cp.async` gather with a loader warp.
- **Question:** for prefill, does TMA with the page id as a coordinate beat the `cp.async` gather at 99 KB with `mma.sync` consumers?
- **Design constraint:** the page size must be a multiple of the KV tile. Page 64 with a 64-row tile, or page 128 with sub-tile addressing as b12x decode does.
- **Reference for cost:** FA #2348 reports paged at 1.09–1.18× the dense time on GB10 with `cp.async`; FA #2980 reports paged FP8 at 0.883× of contiguous.
- **Gate:** phase-1 stress test, with attention to the fence question of section 8.3.

### 11.9 Phase 6 — Native FP8 in the paths the engines call

- **Finding:** native FP8 multiply exists in b12x (opt-in), FA #2980 and #3132 (open), SGLang #45871 (open). vLLM's `FLASHINFER` path converts FP8 to 16-bit. Reported gains are 1.7–2.0×, all on RTX PRO 6000.
- **Experiments:**
  - BF16 KV, FP8 converted, FP8 native, on both machines.
  - `B12X_TURBO_ATTN=1` against the default, with the batch and chunk thresholds that limit it for decode.
  - Accuracy against the FP32 oracle and a task metric, reported separately from speed.
- **Why the Spark matters here:** if hypothesis A holds, converted FP8 already halves the bytes read, and native FP8 adds little for decode. If native FP8 still helps on the Spark, decode there is not purely bandwidth-bound.
- **Technique to note:** both #3132 and #45871 scale P by 2⁸ before the FP8 cast so that probabilities use the format's range.

### 11.10 Phase 7 — Coverage gaps

| Gap | Where | Possible work |
|---|---|---|
| Head dimension 64 in b12x decode | trait rule plus merge restriction (section 4.5) | allow `num_mma_kv = 2` for 16-row tiles; extend the merge |
| Head dimension 512 | SGLang `fa4_sm120` raises; b12x supports 128/192/256; FMHAv2 handles it | a 512 tile within 101,376 B needs FP8 operands or a narrower tile |
| BF16 paged KV in upstream FA4 | #2348 and #3060 open | review and test on both machines |
| Multimodal prefix | composites are major 9 or 10 only | measure FlashInfer's native bidirectional wrapper against Triton |
| Decode context parallelism | b12x and XQA lack LSE output | relevant for 2× RTX PRO 6000 or 2× Spark |

### 11.11 Phase 8 — Dispatch

- **Finding:** vLLM picks by validity. Its SM12x default is FA2, and b12x is unreachable without a flag.
- **Deliverable:** a rule per machine, phase and KV dtype, backed by phase-2 tables. SGLang's per-phase flags can run such a rule today. vLLM needs it in `_get_backend_priorities`.
- **A caution from the data:** #2980's BF16 decode was slower than FA2 in 47 of 108 shapes. A blanket "prefer the new kernel" rule would regress nearly half of them.

### 11.12 Order

Phases 0–2 are prerequisites. After that the order depends on which hypothesis survives:

- **A holds (Spark is bandwidth-bound):** phase 6 first on the Spark, then phase 4.
- **B holds (RTX PRO 6000 is occupancy-bound):** phases 3 and 5 first on the RTX PRO 6000.

---

## 12. Operator reference

### 12.1 vLLM on SM12x

| Goal | Flags | Expected log | Preconditions |
|---|---|---|---|
| FA2 | `--attention-backend FLASH_ATTN` | `Using FlashAttention version 2` | no sinks, no FP8/NVFP4 KV, no multimodal prefix |
| FlashInfer with XQA decode | `--attention-backend FLASHINFER` | `decode_backend=xqa ... arch=sm120` or `sm121` | cubins reachable; head dim 16–256 |
| FlashInfer, XQA off | add `--attention-config.use_trtllm_attention=0` | `decode_backend=flashinfer-native` | sinks become unsupported |
| b12x | `--attention-backend b12x` | `Using b12x with q_heads=...` | BF16 model; head dim 128/192/256; block size unset, 64 or 128 |
| Triton | `--attention-backend TRITON_ATTN` | `Using AttentionBackendEnum.TRITON_ATTN backend.` | none |
| Sparse MLA | `--attention-backend FLASHINFER_MLA_SPARSE_SM120` | same pattern | `index_topk == 2048`, BF16 |

Environment variables:

| Variable | Effect on SM12x | Citation |
|---|---|---|
| `VLLM_HAS_FLASHINFER_CUBIN=1` | skips the network probe that gates XQA | `vllm/utils/flashinfer.py:85-92` |
| `VLLM_BATCH_INVARIANT=1` | disables XQA; `FLASHINFER` and `B12X` fail validation | `utils/flashinfer.py:875-877`; `backend.py:221-223` |
| `VLLM_KV_CACHE_LAYOUT` | overrides the cache layout | `vllm/envs.py:270-273` |
| `B12X_TURBO_ATTN=1` | native FP8 multiply in b12x (read by FlashInfer) | `flashinfer/.../paged/_forward.py:78-79` |

### 12.2 SGLang on SM12x

```bash
# FA4, both phases. BF16/FP16 KV only; head dim <= 256; not for MLA or hybrid GDN.
# Page size is not forced on SM12x; 64 enables the page-TMA head-dim-256 decode kernel.
--attention-backend fa4 --page-size 64

# FA4 prefill + XQA decode
--prefill-attention-backend fa4 --decode-attention-backend trtllm_mha --page-size 64

# FMHAv2 prefill + XQA decode. Not with --kv-cache-dtype fp8_e4m3.
--attention-backend trtllm_mha --page-size 64

# FP8 KV with XQA decode
--prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha \
  --kv-cache-dtype fp8_e4m3 --page-size 64

# NVFP4 KV: pass only the dtype; the resolver picks flashinfer prefill and XQA decode
--kv-cache-dtype nvfp4
```

Debug aid: `FLASH_ATTENTION_ARCH=sm_120` overrides FA4's architecture detection [SRC] `flash_attn/cute/interface.py:86-101`.

### 12.3 Verification commands

Each falsifies one claim in this report if its output differs.

| Claim | Command | Expected |
|---|---|---|
| vLLM default is FA2 | `vllm serve <bf16 model>`, grep the log | `Using FLASH_ATTN attention backend out of potential backends` and `Using FlashAttention version 2` |
| FP8 KV moves vLLM to FlashInfer + XQA | add `--kv-cache-dtype fp8` | backend `FLASHINFER`; `decode_backend=xqa` |
| b12x is installed | `python -c "import importlib.util as u; print(u.find_spec('b12x'))"` | not `None` |
| Fork gates FA4 off | `python -c "from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported as f; print(f(4), f(2))"` | `False True` |
| Shared-memory capacity constant | `python -c "import cutlass.utils as u; print(u.get_smem_capacity_in_bytes('sm_120'))"` | `101376` |
| SGLang helper semantics on a Spark | `python -c "from sglang.srt.utils.device import is_sm120, is_sm120_supported, is_sm121; print(is_sm120(), is_sm120_supported(), is_sm121())"` | `False True True` |
| b12x head dimension 64 mechanism | `select_paged_forward_traits(cta_tile_q=16, head_dim_qk=64, head_dim_vo=64, ...)` | `ValueError: no valid NUM_MMA_KV...`; success with `exact_num_mma_kv=2` |
| Upstream FA4 head dimension 256 | call `flash_attn.cute.interface.flash_attn_func` with BF16, head dim 256 | a launch-time shared-memory error |

---

## 13. Corrections, disagreements between sources, and unknowns

### 13.1 Corrections to the previous version of this report

| Previous statement | Correct statement |
|---|---|
| "No public measurement was found comparing any two of these kernels" | Pull requests report measurements on both machines (section 9). None is independent. |
| The research path proposed page-aligned TMA tiles as a new idea | b12x decode and one SGLang specialization already do it (sections 4.5, 4.7) |
| The research path proposed split-KV for SM12x as missing | It exists in FA2, FlashInfer fa2, XQA, b12x decode and SGLang `fa4_sm120`; it is missing only in upstream FA4 |
| vLLM's default on SM12x is FA2 | True for plain BF16 models; `FLASHINFER` for FP8/NVFP4 KV, sinks, or a multimodal prefix |
| Only the 99 KB shared-memory limit was given | 99 KB is per block; per SM is 128 KB or 100 KB depending on the NVIDIA document |
| Forcing FA4 in vLLM "depends on the fork, which was not inspected" | The gate is in vLLM's tree and excludes SM12x; the expected result is a startup failure |
| vLLM PR #50862 was cited as an FA4 gate fix | It is closed without merging and concerns GDN prefill |

### 13.2 Documentation that lags the code

- **vLLM docs** call XQA "the SM90 decode path" and say the FlashAttention default is "FA2 otherwise"; the code enables XQA on the SM12x family [SRC] `docs/design/attention_backends.md:176-178`.
- **vLLM docs** list FlashInfer capability as 8.x–9.x, 9.0 and 10.x; the source accepts up to 12.1.
- **vLLM docs** list b12x KV dtypes as `auto` and capability "Any"; the source accepts BF16 and FP8 and exactly (12, 0) and (12, 1).
- **SGLang docs** list FA4's native page size as 128 with no qualifier; the forcing is SM100-only.
- **SGLang docs** say Blackwell defaults to `trtllm_mha`; that is SM100 only.
- **SGLang docs** say `trtllm_mha` is decode-only for hybrid GDN models on SM120; the assertion allows prefill.
- **SGLang docstring** in `get_default_attn_backend` says "trtllm_mha does not support SM120"; the validation code accepts it.
- **The author's vault note** "Writing FlashInfer Kernels for DGX Spark GB10" says "FA4 is tcgen05-only → SM100 only". Upstream has had an `mma.sync` SM120 path since March 2026.

### 13.3 Disagreements between primary sources

- **Shared memory per SM, capability 12:** 128 KB (tuning guide) against 100 KB (programming guide).
- **Thread blocks per SM, capability 12:** 32 against 24, same two documents.
- **Minimum toolkit for DGX Spark:** CUTLASS's README says CUDA 13.0; its `config.h` enables SM121 from 12.9.
- **FA #2998's body** says `can_implement` rejects head dimension 256; no call to `can_implement` was found on the forward path at `9f9742c`.
- **FA #3109's body** describes main's SM120 path as "fixed 64 × 64 tiles"; that matches the backward kernel, not the forward.

### 13.4 Not determined

- SM counts read from a device (188 and 48 are derived).
- Any limit specific to capability 12.1; neither NVIDIA guide mentions it.
- Whether the `flashinfer-python==0.7.4` wheel contains `b12x.attention.paged`. FlashInfer's own guidance says experimental features are JIT-only and must not ship in pre-built packages.
- FMHAv2's staging mechanism.
- Whether b12x paged attention has the #7675 ordering hazard.
- Whether SGLang's dense `trtllm_mla` runs on SM12x.
- Whether `fa4` prefill with `trtllm_mha` decode is numerically correct end to end.
- The contents of SGLang #49044.
- The fleet's current FlashInfer versions (0.6.16.post3 and 0.7.0 as of 2026-10-10 [VAULT]).
- Maintainer positions on any open FlashAttention pull request except #2336 and #2348.

### 13.5 The smallest experiment that would change the plan

On each machine, serve one BF16 head-dimension-128 model twice: with no flag (FA2), and with `--attention-backend b12x`. Confirm each backend from the log lines in section 5.5. Measure batch-1 inter-token latency at a 32K context.

- **Near-equal on the Spark, clearly different on the RTX PRO 6000:** hypotheses A and B both stand. Do phase 6 first on the Spark and phases 3 and 5 first on the RTX PRO 6000.
- **b12x clearly faster on the Spark too:** decode there is not purely bandwidth-bound. Phase 4 (Spark tuning) moves ahead of phase 6.
- **FA2 faster on the RTX PRO 6000:** consistent with #2980's finding that new BF16 kernels lose to FA2 on many decode shapes, and a reason to treat phase 8 (dispatch) as the main deliverable.

---

## 14. Sources

All accessed 2026-11-17.

**NVIDIA documentation**
- [S1] PTX ISA manual, version 9.5 — https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- [S2] Blackwell tuning guide, version 13.5 — https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- [S3] CUDA programming guide, compute capabilities — https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- [S4] CUDA GPU compute capability list — https://developer.nvidia.com/cuda/gpus
- [S5] DGX Spark hardware overview — https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- [S6] RTX PRO 6000 Blackwell datasheet (hosted by PNY; seen as a search snippet, not opened) — https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/NVIDIA-RTX-PRO-6000-Blackwell-Workstation-Edition.pdf
- [S7] RTX PRO 6000 Blackwell product page — https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/

**CUTLASS** (`b2f249a`)
- [S8] https://github.com/NVIDIA/cutlass — files cited: `include/cutlass/arch/config.h`, `include/cutlass/arch/arch.h`, `include/cutlass/arch/barrier.h`, `include/cute/arch/mma_sm120.hpp`, `include/cute/arch/config.hpp`, `include/cute/arch/copy_sm90_tma.hpp`, `include/cutlass/gemm/collective/sm120_mma_tma.hpp`, `include/cutlass/gemm/collective/builders/sm120_mma_builder.inl`, `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`, `include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`, `README.md`

**FlashAttention**
- [S9] Upstream repository (`9f9742c`) — https://github.com/Dao-AILab/flash-attention — files cited: `flash_attn/cute/flash_fwd_sm120.py`, `flash_fwd.py`, `interface.py`, `config.py`, `paged_kv.py`, `flash_fwd_sm90.py`, `flash_fwd_sm100.py`, `flash_fwd_combine.py`, `csrc/flash_attn/flash_api.cpp`, `csrc/flash_attn/src/flash_fwd_launch_template.h`, `kernel_traits.h`, `flash_fwd_kernel.h`
- [S10] Issue #2307 "FA4 SM120 support?" (open) — https://github.com/Dao-AILab/flash-attention/issues/2307
- [S11] Issue #2996 (closed 2026-10-23) — https://github.com/Dao-AILab/flash-attention/issues/2996
- [S12] PRs #2329 (merged), #2336, #2348, #2980 (open) — https://github.com/Dao-AILab/flash-attention/pull/2329 , `/pull/2336` , `/pull/2348` , `/pull/2980`
- [S13] PRs #2998, #3026, #3060, #3108, #3109, #3132 (open) — https://github.com/Dao-AILab/flash-attention/pull/3109 and siblings
- [S14] vLLM fork (`8dd9eaea`) — https://github.com/vllm-project/flash-attention — files cited: `vllm_flash_attn/flash_attn_interface.py`, `csrc/flash_attn/flash_api.cpp`, `csrc/flash_attn/src/kernel_traits.h`

**FlashInfer** (`778f57e`)
- [S15] Repository — https://github.com/flashinfer-ai/flashinfer — files cited: `flashinfer/experimental/b12x/attention/paged/{forward_paged,forward_extend_generic,planner,traits,merge,_forward,__init__,api}.py`, `flashinfer/experimental/b12x/_lib/{gating,intrinsics,smem}.py`, `flashinfer/utils.py`, `flashinfer/prefill.py`, `flashinfer/decode.py`, `flashinfer/xqa.py`, `flashinfer/jit/xqa.py`, `flashinfer/compilation_context.py`, `include/flashinfer/attention/{prefill,decode,scheduler}.cuh`, `include/flashinfer/utils.cuh`, `csrc/xqa/{mha.cu,mla_sm120.cu,utils.cuh,mma.cuh,ldgsts.cuh,mhaUtils.cuh}`, `flashinfer/experimental/AGENTS.md`
- [S16] Release notes — https://flashinfer.ai/releases/
- [S17] Issue #3170, DGX Spark support audit (open) — https://github.com/flashinfer-ai/flashinfer/issues/3170
- [S18] Issue #7675, b12x GEMM TMA ordering (open) — https://github.com/flashinfer-ai/flashinfer/issues/7675
- [S19] PR #7676, fence fix (open) — https://github.com/flashinfer-ai/flashinfer/pull/7676
- [S20] Issue #6977, b12x head dimension 64 (open) — https://github.com/flashinfer-ai/flashinfer/issues/6977
- [S21] b12x repository (archived 2026-10-06) — https://github.com/local-inference-lab/b12x
- [S22] PR #5767, b12x migration (merged 2026-10-06) — https://github.com/flashinfer-ai/flashinfer/pull/5767

**vLLM** (`f39d833`)
- [S23] Repository — https://github.com/vllm-project/vllm — files cited: `vllm/platforms/cuda.py`, `vllm/platforms/interface.py`, `vllm/v1/attention/selector.py`, `vllm/v1/attention/backend.py`, `vllm/v1/attention/backends/{flash_attn,fa_utils,flashinfer,b12x,triton_attn,triton_flash_attn,triton_flashinfer,composite}.py`, `vllm/v1/attention/backends/mla/`, `vllm/utils/{flashinfer,b12x}.py`, `vllm/vllm_flash_attn/flash_attn_interface.py`, `vllm/config/attention.py`, `vllm/engine/arg_utils.py`, `requirements/cuda.txt`, `docs/design/attention_backends.md`
- [S24] Attention backends documentation — https://docs.vllm.ai/en/latest/design/attention_backends/

**SGLang** (`b2cc023f`)
- [S25] Repository — https://github.com/sgl-project/sglang — files cited: `python/sglang/kernels/ops/attention/fa4_sm120/{flash_fwd,flash_fwd_decode,runtime,dispatch,policy,paged_kv,scheduler}.py`, `python/sglang/kernels/ops/attention/flash_attention_v4_sm120.py`, `flash_attn/cute/interface.py`, `flash_mla_sm120.py`, `python/sglang/srt/arg_groups/{model_override_base,overrides,attention_hook,kv_cache_hook,speculative_hook,choices}.py`, `python/sglang/srt/layers/attention/{flashattention_backend,flashinfer_backend,trtllm_mha_backend,attention_registry}.py`, `python/sglang/srt/utils/device.py`, `python/sglang/srt/runtime_context.py`, `python/pyproject.toml`
- [S26] Attention backend documentation — https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/attention_backend.mdx
- [S27] PR #32991, SM12x FA4 kernels (merged 2026-08-13) — https://github.com/sgl-project/sglang/pull/32991
- [S28] PR #45871, native FP8 KV (open) — https://github.com/sgl-project/sglang/pull/45871
- [S29] PR #45262 (merged 2026-11-10) — https://github.com/sgl-project/sglang/pull/45262
- [S30] PR #35899 (closed, not merged) — https://github.com/sgl-project/sglang/pull/35899
- [S31] Issue #50749 (open) — https://github.com/sgl-project/sglang/issues/50749
- [S32] Issue #40286 (open) — https://github.com/sgl-project/sglang/issues/40286
- [S33] Issue #42364 (open) — https://github.com/sgl-project/sglang/issues/42364
- [S34] Issue #33422 (open) — https://github.com/sgl-project/sglang/issues/33422

**Author's notes**
- "FlashInfer SM12x Release Tracker" (reviewed 2026-10-10); "Writing FlashInfer Kernels for DGX Spark GB10" (status dated 2026-09); "FlashInfer Issue 3013 - CUTLASS Slow on SM120-121" (reviewed 2026-10-11); `consumer-gpu-kernel-guide` tuning notes.

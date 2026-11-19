# Dossier: how vLLM serves attention on compute capability 12.0 / 12.1

**Source:** vLLM `main` at commit `f39d833` (2026-11-17), read from the local checkout. All citations are `vllm/relative/path:line` against that commit.

**Method and evidence labels.** Nothing in this dossier was executed: no vLLM process was started and no GPU was involved. Every statement is therefore one of two kinds, and is labelled where the distinction matters:

- **[SRC]** a source-code fact: the cited lines say exactly this.
- **[INF]** an inference: a consequence derived by following several source facts, not observed at runtime.

"SM12x" below means CUDA compute capability major 12, i.e. 12.0 (RTX PRO 6000 Blackwell, RTX 5090) and 12.1 (DGX Spark GB10). vLLM's family test is `Platform.is_device_capability_family(120)`, which is `(current_capability.to_int() // 10) == (capability // 10)` (`vllm/platforms/interface.py:587-598`), so it is true for both 12.0 and 12.1. `is_device_capability((12, 0))` is an exact match and is false on 12.1 (`vllm/platforms/interface.py:564-580`).

**Headline conclusions (each is expanded and cited below):**

1. On SM12x the automatic choice for standard attention is `FLASH_ATTN` running FlashAttention 2, unless the configuration needs FP8/NVFP4 KV cache, attention sinks, or a multimodal prefix; each of those three makes `FLASH_ATTN` fail validation, and `FLASHINFER` (priority 2) takes over. [INF from SRC]
2. `FLASHINFER` on SM12x is a two-kernel backend: FlashInfer's native prefill wrapper with `backend="auto"`, and the XQA decode kernel for decode, the latter only if FlashInfer's prebuilt cubins are reachable. [SRC]
3. `B12X` is a real, complete backend for (12,0) and (12,1) but appears in no priority list, so it runs only when requested by name. [SRC]
4. The premise "the `is_fa_version_supported` implementation lives in the bundled wheel" is not accurate for this commit: the Python gate is in-tree at `vllm/vllm_flash_attn/flash_attn_interface.py`, and it excludes family 120 from FA4 explicitly. [SRC]
5. `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` does not occur anywhere in the repository. [SRC: `grep -rIl` over the whole checkout returned no file]

---

## 1. Selection walk-through

### 1.1 From CLI to selector

- `--attention-backend` is registered at `vllm/engine/arg_utils.py:1121`; `--attention-config` / `-ac` at `vllm/engine/arg_utils.py:1918`. The two are merged at `vllm/engine/arg_utils.py:2764-2774`, and setting both raises `"attention_backend and attention_config.backend are mutually exclusive"`. [SRC]
- `AttentionConfig.backend` is parsed case-insensitively by `AttentionBackendEnum[value.upper()]`, and the string `"auto"` becomes `None` (`vllm/config/attention.py:273-285`). So `b12x`, `B12X`, and `flash_attn` are all accepted spellings. [SRC]
- `get_attn_backend()` (`vllm/v1/attention/selector.py:111-208`) builds an `AttentionSelectorConfig`. Two details matter operationally: `block_size` is passed only when the user set `--block-size` (`selector.py:137-140`), and `use_batch_invariant` comes from `envs.VLLM_BATCH_INVARIANT` (`selector.py:173`). A per-KV-group override `backend_per_kind` wins over the global backend (`selector.py:189-197`). [SRC]
- `_cached_get_attn_backend` calls `current_platform.get_attn_backend_cls(...)` (`selector.py:211-233`). [SRC]

### 1.2 `CudaPlatformBase.get_attn_backend_cls`

If a backend was named, only that backend is validated; failure raises, success logs and returns (`vllm/platforms/cuda.py:471-490`):

```python
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return _backend_cls_path(backend_class)
```

Otherwise `get_valid_backends` (`cuda.py:417-458`) walks `_get_backend_priorities(...)` and the lowest-index valid backend wins (`cuda.py:520-523`).

**Verified priority lists for major 12** (`vllm/platforms/cuda.py:131-135` and `166-178`): MLA is `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]`; standard attention falls into the generic `else` and is `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT]`. The `TRITON_FLASH_ATTN` composite is prepended only when `device_capability.major == 9 and use_mm_prefix` (`cuda.py:168-172`), and `TRITON_FLASHINFER` only in the `major == 10` branch (`cuda.py:157-159`). [SRC]

### 1.3 Log lines an operator can grep

| Situation | Format string | Location |
|---|---|---|
| Backend forced by name | `"Using %s backend."` (argument is the enum, so it prints e.g. `AttentionBackendEnum.B12X`) | `cuda.py:489` |
| Auto-selected | `"Using %s attention backend out of potential backends: %s."` | `cuda.py:549-557` |
| Auto, rejected backends (DEBUG only) | `f"Some attention backends are not valid for {cls.device_name} with {config_str}. Reasons: {reasons_str}."` | `cuda.py:508-511` |
| No valid backend | `f"No valid attention backend found for {cls.device_name} with {config_str}. Reasons: {reasons_str}."` | `cuda.py:513-516` |
| `--block-size` knocked out a better backend | `"--block-size %d precluded higher-priority backend(s) %s. Using %s instead, ..."` | `cuda.py:539-547` |
| FlashAttention version | `"Using FlashAttention version %s"` | `flash_attn.py:1083-1086` |
| FlashInfer decode kernel and Q dtypes | `"FlashInfer resolved query dtypes: prefill=%s, decode=%s, decode_backend=%s, kv_cache_dtype=%s, arch=%s"` | `flashinfer.py:1216-1224` |
| b12x plan summary | `"Using b12x with q_heads=%d kv_heads=%d head_dim_qk=%d head_dim_vo=%d window_left=%d planned_page_sizes=%s verify_q_per_req=%d extend_q_capacities=%s scratch=%d bytes."` | `b12x.py:718-731` |
| MLA prefill backend | `"Using %s MLA prefill backend."` | `mla/prefill/selector.py:106,131` |

In the FlashInfer line, `decode_backend` is `xqa`, `trtllm-gen`, or `flashinfer-native`, and `arch` is built as `f"sm{capability.major}{capability.minor}"`, so it prints `sm120` or `sm121` (`flashinfer.py:1209-1215`). [SRC]

### 1.4 Every check in `AttentionBackend.validate_configuration`

`vllm/v1/attention/backend.py:296-389`. Each failing check appends one string; an empty list means valid. In order:

| # | Predicate | Reason string |
|---|---|---|
| 1 | `not supports_head_size(head_size)` | `head_size not supported` |
| 2 | `not supports_dtype(dtype)` | `dtype not supported` |
| 3 | `not supports_kv_cache_dtype(kv_cache_dtype)` | `kv_cache_dtype not supported` |
| 4 | `not supports_block_size(block_size)` | `block_size not supported` |
| 5 | `use_mm_prefix and not supports_mm_prefix()` | `partial multimodal token full attention not supported` |
| 6 | `use_mla != is_mla()` | `MLA not supported` / `non-MLA not supported` |
| 7 | `has_sink and not supports_sink()` | `attention sinks not supported` |
| 8 | `use_sparse != is_sparse()` | `sparse not supported` / `non-sparse not supported` |
| 9 | `use_per_head_quant_scales and not ...` | `per-head quant scales not supported` |
| 10 | `not supports_compute_capability(cap)` | `compute capability not supported` |
| 11 | `not supports_attn_type(attn_type)` | `attention type {attn_type} not supported` |
| 12 | `has_sliding_window and not ...` | `sliding window not supported` |
| 13 | `use_rswa and not supports_rswa()` | `R-SWA not supported` |
| 14 | `use_non_causal and not ...` | `non-causal attention not supported` |
| 15 | `use_mla and use_non_causal and use_dcp and not supports_non_causal_dcp()` | `non-causal MLA attention with DCP not supported` |
| 16 | `use_batch_invariant and not ...` | `batch invariance not supported` |
| 17 | `use_kv_connector and not ...` | `KV connector not supported` |
| 18 | `use_hisparse and not ...` | `HiSparse not supported` |
| 19 | `use_pcp and not supports_pcp()` | `PCP not supported` |
| 20 | `use_dcp and not supports_dcp()` | `DCP not supported` |
| 21 | `use_adaptive_verification and not supports_device_cpu_query_lens_mismatch()` | `device-cpu query lens mismatch not supported, this is needed for adaptive verification` |
| 22 | `supports_combination(...)` returns a string | that string verbatim |

Base-class defaults that decide SM12x outcomes: `supports_block_size(None)` is `True`, and otherwise a block size passes if it is a multiple of any advertised kernel size (`backend.py:134-151`); `supports_batch_invariance()` defaults to `False` (`backend.py:221-223`); `supports_dcp()` reads the impl class attribute (`backend.py:255-260`). [SRC]

---

## 2. `FLASH_ATTN` on major 12

### 2.1 Resulting version: FA2

`get_flash_attn_version` (`vllm/v1/attention/backends/fa_utils.py:91-100`):

```python
        if device_capability.major == 9 and is_fa_version_supported(3):
            # Hopper (SM90): prefer FA3
            fa_version = 3
        elif device_capability.major == 10 and is_fa_version_supported(4):
            # Blackwell (SM100+, restrict to SM100 for now): prefer FA4
            fa_version = 4
        else:
            # Fallback to FA2
            fa_version = 2
```

Major 12 takes the `else`. [SRC] FA2 availability is `_vllm_fa2_C` importable and `has_device_capability(80)` (`vllm/vllm_flash_attn/flash_attn_interface.py:52-59`).

### 2.2 What the FA2 result does to validation on SM12x

- **Sinks: not served.** `supports_sink()` returns `flash_attn_supports_sinks()` (`flash_attn.py:422-426`), which is `get_flash_attn_version() in (3, 4)` (`fa_utils.py:244-247`). With FA2 that is `False`, so check 7 emits `attention sinks not supported`. The separate predicate `has_sink and device_capability < DeviceCapability(9, 0)` → `"sink not supported on compute capability < 9.0"` (`flash_attn.py:455-456`) does **not** fire on 12.x, because 12.x is not below 9.0; the rejection comes from `supports_sink`, not from that line. If the gate were bypassed, the FA2 dispatch itself raises `NotImplementedError("FA2 does not support s_aux")` (`flash_attn_interface.py:369-370`), and `FlashAttentionImpl.__init__` asserts `"Sinks are only supported in FlashAttention 3"` (`flash_attn.py:1105-1108`). **FA2 cannot serve sinks on 12.x.** [SRC + INF]
- **FP8 KV cache: not served.** `flash_attn_supports_kv_cache_dtype` returns true only for `(FA3 or d512 FA4) and SM90`, or `FA4 and family(100)` (`fa_utils.py:231-237`). On 12.x both arms are false, so check 3 fails, and `supports_combination` adds `"FP8 KV requires FA3 (SM90), equal-d512 FA4 (SM90), or FA4 (SM100)."` (`flash_attn.py:464-474`). [SRC]
- **Multimodal prefix: not served.** `supports_mm_prefix()` is `is_fa_version_supported(4)` (`flash_attn.py:418-420`), false on 12.x (section 2.5). [SRC]
- **Head size:** `head_size % 8 == 0` and `<= 256`; the `<= 512` extension requires `is_fa_version_supported(4)` (`flash_attn.py:395-406`). [SRC]
- **Block size:** `[MultipleOf(16)]` (`flash_attn.py:332-340`); the only fixed-size case is SM90 FA4 FP8 d512. [SRC]
- **Compute capability:** `capability >= DeviceCapability(8, 0)` (`flash_attn.py:428-430`). [SRC]
- **Supported unconditionally:** sliding window, R-SWA, non-causal, batch invariance, all four attention types (`flash_attn.py:356-380`); DCP via `FlashAttentionImpl.supports_dcp = True` (`flash_attn.py:1038`). [SRC]

### 2.3 Arguments passed to `flash_attn_varlen_func`

vLLM's FlashAttention backend does not split prefill from decode: one call serves the whole batch (`flash_attn.py:1378-1403`), dispatched through `_FA4_DENSE_ATTENTION_KERNEL.__call__`, which simply forwards to `flash_attn_varlen_func` (`flash_attn.py:109-112, 263-271`).

```python
                softmax_scale=self.scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                dynamic_causal=dynamic_causal,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
```

Values on SM12x with FA2 [INF from the cited SRC]:

- `fa_version=2`.
- `block_table` = the batch block table (paged KV); `k`/`v` are views of the cache produced by `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` (`flash_attn.py:1241`).
- `scheduler_metadata=None`, because `self.aot_schedule = fa_version == 3` (`flash_attn.py:687`).
- `num_splits=0`. `self.max_num_splits` is raised to `flash_attn_max_num_splits_for_cuda_graph` only inside `if self.use_full_cuda_graph and self.aot_schedule` (`flash_attn.py:729-749`), which is FA3-only. Under `VLLM_BATCH_INVARIANT` the value becomes 1 (`flash_attn.py:846-847`). FA2 rejects `num_splits > 1` (`flash_attn_interface.py:371-372`).
- `s_aux=None` (sinks are excluded upstream), `mask_mod=None`, `aux_tensors=None`; the mm-prefix and R-SWA mask paths are guarded by `self.vllm_flash_attn_version == 4` (`flash_attn.py:1315, 1354`).
- Per-sequence (tensor) `causal` raises `"Per-sequence causal requires FA4. Current version: FA2"` (`flash_attn.py:1366-1371`).

### 2.4 CUDA graph support

`flash_attn.py:593-597`:

```python
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() in (3, 4) or current_platform.is_xpu()
        else AttentionCGSupport.UNIFORM_BATCH
    )
```

FA2 on 12.x therefore reports `UNIFORM_BATCH`. The preceding comment (`flash_attn.py:578-592`) explains the reason: FA2 has special packed-GQA handling for `max_query_len=1`, so a graph captured for single-token decode does not cover mixed prefill-decode. [SRC] Note that this is evaluated once at class-definition time.

### 2.5 Forcing `--attention-config.flash_attn_version=4` on major 12

The gate is in-tree (`vllm/vllm_flash_attn/flash_attn_interface.py:72-86`):

```python
def _is_fa4_supported() -> tuple[bool, str | None]:
    if not FA4_AVAILABLE:
        return False, f"FA4 is unavailable due to: {FA4_UNAVAILABLE_REASON}"
    from vllm.platforms import current_platform

    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(110)
    ):
        return (
            False,
            "FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x",
        )
    return True, None
```

Chain through `get_flash_attn_version` when the config value is 4 on 12.x [SRC for each step, INF for the end-to-end outcome]:

1. The config override sets `fa_version = 4` (`fa_utils.py:102-110`).
2. If `VLLM_BATCH_INVARIANT` is set, the version drops to 2 (`fa_utils.py:164-169`).
3. The TMEM guard applies for `major >= 10`: a `head_size > 128` that is not `256` (with `head_size_v` in `(None, 256)`) or `192/128` is downgraded to FA2 with the warning `"FA4 on Blackwell does not support head_size=%d due to TMEM capacity limits, defaulting to FA version 2."` (`fa_utils.py:171-187`).
4. Otherwise `is_fa_version_supported(4)` is false, so vLLM logs `"FlashAttention %d is not supported (%s); it will not be used."`, then `assert is_fa_version_supported(fa_version)` fails, the `AssertionError` is caught, and the function returns `None` (`fa_utils.py:189-199`).

Consequence of `None` [INF]: nothing in `validate_configuration` requires a non-`None` version, so `FLASH_ATTN` still validates; `FlashAttentionImpl.__init__` logs `Using FlashAttention version None`; and the first `forward` fails on `assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected."` (`flash_attn.py:1201-1203`), which precedes the profiling-run early return. For the common head sizes 64, 128 and 256, forcing FA4 on 12.x is therefore expected to produce a startup failure rather than a silent FA2 fallback. This was not run.

---

## 3. `FLASHINFER` on SM12x

### 3.1 Every branch keyed on family 120 / major 12

| Location | Meaning |
|---|---|
| `flashinfer.py:155-156` | `_nvfp4_kv_on_fa2()` true for families 80, 90, 120: NVFP4 KV is read by fa2 kernels, not trtllm-gen |
| `flashinfer.py:754` | `supports_combination`: NVFP4 on majors 8/9/12 rejects `head_size == 64` and sinks |
| `flashinfer.py:779-781` | capability window `>= (8,0)` and `<= (12,1)` |
| `flashinfer.py:793-796` | `supports_sink`: SM90 or family 120 → needs XQA decode only |
| `flashinfer.py:1089-1091` | XQA may read an NVFP4 cache only on family 120 |
| `flashinfer.py:1278-1287` | decode Q dtype = model dtype with FP8 KV (XQA) |
| `flashinfer.py:1326-1328` | `is_xqa_arch` for CUDA graph level |
| `flashinfer.py:1341-1342` | NVFP4 does not remove uniform-batch support on family 120 |
| `flashinfer.py:1429-1432` | decode kernel selector returns `XQA` |
| `flashinfer.py:1523-1526, 1558-1561` | sink prefill wrapper chosen on SM90 / family 120 |
| `flashinfer.py:1387-1400` | workspace sizing comment naming `sm_120` (no predicate; applies to all archs) |
| `vllm/utils/flashinfer.py:884-887` | `supports_trtllm_attention`: SM90 / family 120 → decode only |
| `vllm/utils/flashinfer.py:987-992` | auto-detect prefers XQA with FP8 KV (prefill-side helper; see 3.3) |

### 3.2 Prefill on SM12x

`prefill_use_trtllm` is `causal and use_trtllm_attention(..., is_prefill=True, ...)` (`flashinfer.py:1765-1777`), and `supports_trtllm_attention(is_prefill=True)` returns `not is_prefill` = `False` on family 120 (`vllm/utils/flashinfer.py:884-887`). So prefill always takes FlashInfer's native wrapper [INF from SRC]. `_get_prefill_wrapper` (`flashinfer.py:1573-1581`):

```python
                    # NVFP4 KV cache requires the trtllm-gen backend inside
                    # the wrapper on SM100; SM8x, SM90 and SM12x read it with fa2.
                    backend = "trtllm-gen" if self.nvfp4_trtllm else "auto"
                    self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                        self._get_workspace_buffer(),
                        get_flashinfer_layout_string(self.kv_cache_layout),
                        backend=backend,
                    )
```

On SM12x `nvfp4_trtllm` is always false, so the wrapper is `BatchPrefillWithPagedKVCacheWrapper(..., backend="auto")`. vLLM does not pin the kernel; which kernel `"auto"` resolves to on SM12x is decided inside FlashInfer and is **not determinable from this repository**. The vLLM comments call the SM12x prefill "fa2-only" (`flashinfer.py:1276-1277, 1294-1295`). With sinks the wrapper is `BatchAttentionWithAttentionSinkWrapper(..., backend="auto", ...)` (`flashinfer.py:1558-1572`); with a multimodal prefix it is `BatchPrefillWithCausalBidirectionalRangesWrapper` (`flashinfer.py:1585-1603`); under DCP it is `BatchDCPPrefillWrapper` (`flashinfer.py:1550-1556`).

### 3.3 Decode on SM12x: XQA versus native

The decode kernel is fixed statically in the metadata builder, not per batch (`flashinfer.py:1086-1104`). XQA is used when **all** of the following hold [SRC]:

1. `can_use_trtllm_attention(num_qo_heads, num_kv_heads, is_prefill=False)` (`vllm/utils/flashinfer.py:906-914`), i.e.
   - `--attention-config.use_trtllm_attention` is not `False` (`force_use_trtllm_attention()`, `utils/flashinfer.py:893-903`);
   - `supports_trtllm_attention(False)`: `VLLM_BATCH_INVARIANT` unset, **and `has_nvidia_artifactory()` true**, and family 120 (`utils/flashinfer.py:868-890`);
   - `num_qo_heads % num_kv_heads == 0`.
2. `_is_xqa_head_dim_supported(head_dim)`: `16 <= head_dim <= 256 and head_dim % 16 == 0` (`flashinfer.py:840-841`); otherwise the warning `"FlashInfer XQA decode does not support head_dim=%d; reverting this KV-cache group to native FlashInfer decode."` (`flashinfer.py:1111-1121`).
3. DCP is off; otherwise `"FlashInfer XQA decode does not support returning LSE and therefore does not support DCP, reverting to native FlashInfer decode."` (`flashinfer.py:1122-1132`).

`has_nvidia_artifactory()` is true if the `flashinfer_cubin` package is importable or `VLLM_HAS_FLASHINFER_CUBIN=1`, and otherwise depends on an HTTP GET of `FLASHINFER_CUBINS_REPOSITORY` returning 200 within 5 s (`utils/flashinfer.py:61-64, 85-92, 840-864`). On an air-gapped SM12x host without the cubin package, XQA is therefore silently off and sinks become unsupported. [INF from SRC]

The auto-detection thresholds inside `use_trtllm_attention` (`num_tokens <= 256 and kv_cache_dtype == "auto"`, `utils/flashinfer.py:982-1001`) are consulted only for the prefill decision in this backend; the decode decision uses `can_use_trtllm_attention`, which has no token threshold. [SRC: `flashinfer.py:1089-1099, 1778-1780`]

When XQA is off, decode uses `BatchDecodeWithPagedKVCacheWrapper` with `use_tensor_cores=True` and `backend="auto"`, because `_use_fa2_native_decode()` is `not has_device_capability(90)` and is false on 12.x (`flashinfer.py:159-161, 1626-1645`). The explicit `backend="fa2"` string is used only below SM90. [SRC]

The XQA call (`flashinfer.py:2982-3001`) is `flashinfer_xqa_batch_decode_with_kv_cache`, a lazy wrapper for `flashinfer.decode.xqa_batch_decode_with_kv_cache` (`utils/flashinfer.py:353-356`), with `kv_layout`, `window_left`, `sinks`, `q_len_per_req`, `mask`, and `q_cu_seq_lens`.

### 3.4 Q dtype rules

`get_q_data_type` (`flashinfer.py:1264-1305`), SM12x outcomes [INF from SRC]:

| KV cache | Prefill Q | Decode Q |
|---|---|---|
| `auto` | model dtype | model dtype |
| `fp8*` | model dtype ("Architectures with only fa2 (e.g. SM89, SM120) cannot consume FP8 queries", `:1294-1300`) | model dtype (`:1278-1287`, XQA "requires BF16/FP16-Q even with FP8 KV cache") |
| `nvfp4` | model dtype (`nvfp4_fa2`, `:1301-1303`) | model dtype |

`--attention-config.disable_flashinfer_q_quantization` forces the model dtype unconditionally (`:1267-1268`). For XQA with a quantized KV cache, `bmm1_scale = scale * q_scale * k_scale`, where `q_scale` is 1.0 for a non-FP8 query (`flashinfer.py:2382-2392`).

NVFP4 specifics on SM12x: only the literal `"nvfp4"` dtype passes (`"nvfp4_4over6"` is refused because "The slot-mapping writer has no scale search", `flashinfer.py:727-730`); `head_size == 64` and sinks are refused (`:751-759`); DCP raises `"The DCP prefill wrapper cannot read NVFP4 block scales."` (`:1048-1051`); non-causal raises (`:1517-1521`).

### 3.5 Sinks, sliding window, layout, DCP, speculative decode

- **Sinks:** `supports_sink()` on family 120 is `supports_trtllm_attention(is_prefill=False)` and is `False` if `use_trtllm_attention` was forced off (`flashinfer.py:783-796`). Sinks therefore require XQA. They are converted to float32 (`flashinfer.py:2369-2379`). Sinks plus DCP raises (`:1797-1802`).
- **Sliding window:** supported (`:698-700`); the native wrappers need one `window_left` for all layers, else `"Window left is not the same for all layers. One potential fix is to set disable_sliding_window=True"` (`:1804-1808`).
- **KV layout:** on majors other than 10, `supported_kv_cache_layouts()` returns `get_flashinfer_kv_cache_layouts()` (`:805-814`), the default preference filtered to layouts FlashInfer can name; `LBNHC` (FlashInfer "NHD") is first (`backends/utils.py:209-246`). The HND assertion applies only to trtllm-gen; the comment says XQA "does not use this SM100 layout gate" (`flashinfer.py:2945-2950`). SM12x therefore defaults to NHD unless another backend in the worker or `VLLM_KV_CACHE_LAYOUT` (`vllm/envs.py:270-273`) changes it. [INF]
- **DCP:** `FlashInferImpl.supports_dcp = True` (`:2233`), with XQA disabled as above.
- **Block sizes:** `[16, 32, 64]`; 128 and larger are advertised only on family 100 (`:595-612`), enforced by an assert (`:1094-1098`).
- **Speculative verify on XQA:** `split_decodes_and_prefills(..., require_uniform=not (self.use_xqa or ...))` (`:1731-1743`) lets ragged multi-token queries stay on the decode path. `_compute_decode_query_lens` returns either a uniform `q_len_per_req` or ragged cumulative offsets (`:1465-1480`), and `_get_decode_mask` builds a packed uint16 draft mask, cached per `(q_len_per_req, causal)` at persistent addresses for graph replay (`:1482-1506, 217-260`). `route_decode = causal or self.use_xqa` (`:1730`) means XQA also handles non-causal draft batches.
- **`VLLM_BATCH_INVARIANT`:** `FlashInferBackend` does not override `supports_batch_invariance`, so check 16 rejects it with `batch invariance not supported` (base default `backend.py:221-223`). [SRC + INF]

### 3.6 CUDA graph levels on SM12x

`get_cudagraph_support` (`flashinfer.py:1309-1352`): DCP → `UNIFORM_SINGLE_TOKEN_DECODE`. Otherwise, since `_use_fa2_native_decode()` is false on 12.x, the result is `UNIFORM_BATCH` when `use_trtllm_attention` is not `False`, `supports_trtllm_attention(False)` is true, and the head dim is XQA-compatible; else `UNIFORM_SINGLE_TOKEN_DECODE`. On an XQA arch, non-causal does not lower the level (`:1349`). [SRC]

---

## 4. `B12X` backend

### 4.1 Gates

- Capability (`b12x.py:216-220`): `(capability.major, capability.minor) in ((12, 0), (12, 1))`.
- Never auto-selected: `AttentionBackendEnum.B12X` (`registry.py:130`) is absent from every list in `_get_backend_priorities` (`cuda.py:81-178`). [SRC]
- Model dtype: `[torch.bfloat16]` only (`b12x.py:170`). KV dtypes: `auto`, `bfloat16`, `fp8`, `fp8_e4m3` (`:54-59`). Head sizes `[128, 192, 256]`, with "head_dim 64 is excluded until flashinfer-ai/flashinfer#6977 is fixed" (`:203-206`).
- Sinks and sliding window: supported (`:208-214`). DCP: `supports_dcp = False` (`:333`). LSE: `can_return_lse_for_decode = False` (`:332`). Batch invariance and mm-prefix: not overridden, so base `False`.
- KV layouts: `(LBHNC, BLHNC)`, i.e. head-major (`:252-254`). `customize_spec` sets `num_head_slots=2` so K and V are separate planes (`:155-168`).

### 4.2 Import and "version gate"

`get_b12x_paged_attention()` returns the module `b12x.attention.paged`: a **top-level `b12x` package**, described as "shipped with flashinfer-python", not a `flashinfer.*` submodule (`vllm/utils/b12x.py:3, 25, 37-48, 80-81`). There is no version comparison in vLLM. The gate is import success plus the module's own `is_supported()` (`b12x.py:245-250`):

```python
        paged_attention = get_b12x_paged_attention()
        if paged_attention is None:
            return "The b12x backend requires flashinfer-python>=0.7.2"
        if not paged_attention.is_supported():
            return "b12x paged attention is not supported on the current device"
```

`requirements/cuda.txt:13` pins `flashinfer-python==0.7.4`. Whether that wheel actually contains `b12x.attention.paged` cannot be confirmed from this repository.

### 4.3 Page sizes

`_B12X_SUPPORTED_PAGE_SIZES = (64, 128)`, preferred 128 (`b12x.py:51-52`); `supports_block_size` accepts only `None`, 64, or 128, with no multiples (`:193-195`). The runtime page size is read from the cache tensor shape (`b12x.py:77-83`):

```python
def _kv_page_size(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return the static kernel page geometry negotiated by vLLM.

    The KV manager can split the configured storage block into a smaller
    kernel page when another backend shares its cache group. Cache shapes are
    fixed before graph capture, so this is not a live-length policy decision.
    """
```

Because the final page size is unknown at `__init__` (`:402-405`), plans are built for **both** page sizes.

### 4.4 Three plan families

All are created through `paged_attention.plan(paged_attention.Caps(..., mode=...))` (`b12x.py:465-498`) and held in dicts (`:679-681`).

| Family | `mode` | Key | Built for | Selected when |
|---|---|---|---|---|
| Decode | `"decode"` | `(page_size, batch)` | each `cudagraph_capture_sizes` entry `<= max_num_seqs`, plus `max_num_seqs` (`:500-504`) | `max_query_len <= 1 and total_q == num_reqs` (`:916`) |
| Verify | `"verify"` | `(page_size, batch)` | every batch `1..max_num_seqs`, only if `1 + num_speculative_tokens > 1` (`:542-548, 687-691`) | `max_query_len == verify_q_per_req and total_q == num_reqs * verify_q_per_req` (`:936-940`) |
| Extend | `"extend"` | `(page_size, batch, q_capacity)` | every batch × every capacity bucket with `batch < q_capacity` (`:692-704`) | everything else (prefill, mixed) |

Extend buckets (`b12x.py:426-437`):

```python
        # Extend dispatch may depend on the static Q tensor capacity, but never
        # on live per-request lengths. Keep a small set of capacity buckets so
        # short/tail prefills do not replay the maximum 8K CTA grid.
        self._extend_q_capacities = tuple(
            sorted(
                {
                    min(max_batched, q_capacity)
                    for q_capacity in (128, 512, 1024, 2048, 4096, max_batched)
                    if q_capacity > 0
                }
            )
        )
```

The bucket is the smallest capacity `>= q_capacity`, where `q_capacity = min(query.shape[0], output.shape[0])` (`:959-976, 1009-1012`). Decode and verify plans may be created lazily, but not during graph capture (`"b12x decode plan was not prepared before CUDA graph capture for page size ..."`, `:921-926`) and not if larger than reserved scratch (`:928-933`). The builder reports `AttentionCGSupport.UNIFORM_BATCH`; its docstring says "Extend/prefill remains eager" (`:269-277`).

**Startup cost [INF]:** the extend-plan count is roughly `2 × max_num_seqs × (number of buckets)`, and verify plans add `2 × max_num_seqs`. Plan construction time is not measurable from source.

### 4.5 Scratch and warmup

`_scratch_nbytes` is the maximum of the decode scratch envelopes and every verify and extend plan's `layout.nbytes` (`:705-709`), reserved once through `current_workspace_manager().get_simultaneous(((self._scratch_nbytes,), torch.uint8))` (`:711-713`) and re-fetched every forward (`:1056-1058`).

`register_cutedsl_warmup_provider(self)` (`:716`) adds the impl to a `WeakSet` (`vllm/model_executor/warmup/cutedsl_warmup.py:35-40`). `get_cutedsl_warmup_compile_units` returns one `CuTeDSLCompileUnit(name="b12x_paged_extend", ...)` per page size, keyed by device, dtypes, head geometry, `window_left`, and whether sinks exist (`b12x.py:819-839`). The compile body binds each extend plan to dummy tensors, calls `paged_attention.compile(binding=...)`, and executes once per batch size (`:733-817`). Only the extend entry is registered; no decode or verify compile unit appears in this file. `b12x_warmup` returns early unless `is_device_capability_family(120)` (`vllm/model_executor/warmup/b12x_warmup.py:51-55`).

### 4.6 Restrictions raised (verbatim)

`__init__` (`b12x.py:349-411`):

- `"b12x does not support ALiBi."`
- `"b12x does not support logits soft cap."`
- `"b12x currently supports decoder self-attention only."`
- `"b12x currently supports only fp8/fp8_e4m3 quantized KV cache dtypes."`
- `"b12x requires q heads divisible by kv heads."`
- `"b12x currently requires canonical softmax scale head_dim**-0.5={expected_scale}, got {scale}."` (`math.isclose`, `rel_tol=1e-5`)
- `"b12x does not yet support decode/prefill context parallelism."` (when `self.total_cp_world_size > 1`)
- `"b12x sinks must have shape [{num_heads}], got {...}."`
- `"b12x currently requires bfloat16 queries."`

`forward` (`:978-1075`): `"b12x does not support fused output quantization."`, `"b12x expected output head dim ..."`, `"b12x plan expects dtype ..."`, `"b12x supports causal attention only."`; from helpers, `"b12x requires runtime page size in (64, 128), got ..."` and `"b12x requires matching K/V page sizes, ..."` (`:91-100`).

### 4.7 FP8 KV, sinks, window

- **FP8:** uint8 cache views are reinterpreted as the platform FP8 dtype (`:892-897`); `layer._k_scale` / `layer._v_scale` must be float32 on the query device and are passed as per-request `k_descale` / `v_descale` (`:853-881, 1071-1072`). The query is never quantized: `supports_quant_query_input = False` (`:715`).
- **Sinks:** kept as float32 and passed as `attention_sink_bias=self.sinks` (`:396, 841-851, 1070`).
- **Sliding window:** `window_left = -1 if sliding_window is None else sliding_window - 1`, baked into capacity queries, plans, and bindings (`:386, 459, 519, 1068`).
- **Cache writes:** `torch.ops._C_cache_ops.reshape_and_cache_flash` (`:1077-1097`).

---

## 5. MLA on major 12

**Auto-selection list:** `[TRITON_MLA, FLASHINFER_MLA_SPARSE_SM120]` (`cuda.py:131-135`). Because check 8 compares `use_sparse` with `is_sparse()`, dense MLA resolves to `TRITON_MLA` and sparse (index_topk) MLA to `FLASHINFER_MLA_SPARSE_SM120`. [INF]

| Backend | Capability gate | Dtypes / KV dtypes | Block sizes | FlashInfer API |
|---|---|---|---|---|
| `TRITON_MLA` | any (`mla/triton_mla.py:163-165`) | fp16, bf16; `auto, float16, bfloat16, fp8, fp8_e4m3` (`:124-131`) | `MultipleOf(16)` (`:137-139`) | none (Triton); `UNIFORM_BATCH` graphs (`:94`); DCP supported (`:182`) |
| `FLASHINFER_MLA_SPARSE_SM120` | `capability.major == 12` (`mla/flashinfer_mla_sparse.py:223-225`) | bf16 only; `auto, fp8, fp8_e4m3, fp8_ds_mla` (`:195-201`) | `MultipleOf(64)` (`:211-213`) | `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` |
| `FLASHINFER_MLA_SPARSE_DSV4` | `capability.major in [10, 12]` (`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:138-140`) | bf16; on major 12 only `fp8, fp8_e4m3, fp8_ds_mla` (`:164-166`) | `[256]` (`:118-120`) | `flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4` |
| `FLASHINFER_MLA_SPARSE_DSV41` | `major in [10, 12]` (`vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py:91`) | on major 12: `fp8, fp8_e4m3, fp8_ds_mla` (`:115-116`) | `[64]` on family 120, else `[128]` (`:66-71`) | same DSV4 call |

Details for `FLASHINFER_MLA_SPARSE_SM120`:

- `supports_combination` requires `has_flashinfer_sparse_mla_sm120()` and `index_topk == 2048`, else `"FLASHINFER_MLA_SPARSE_SM120 requires index_topk=2048; got ..."` (`mla/flashinfer_mla_sparse.py:239-272`). The capability probe only checks that `autotune`, `trtllm_batch_decode_sparse_mla_dsv4`, and `trtllm_batch_decode_with_kv_cache_mla` import and are callable (`vllm/utils/flashinfer.py:481-498`).
- The impl accepts only the packed layout and raises `"FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla KV cache layout; got kv_cache_dtype=..."` (`mla/flashinfer_mla_sparse_sm120.py:70-74`). The apparent conflict with the validation list is resolved by `_canonicalize_sparse_mla_kv_cache_dtype`, which maps `auto`, `fp8`, `fp8_e4m3` to `fp8_ds_mla` for this backend (`vllm/model_executor/layers/attention/mla_attention.py:383-388`). So the KV cache is always FP8-packed, even with `--kv-cache-dtype auto`. [SRC + INF]
- It refuses `alibi_slopes / sliding_window / logits_soft_cap` (`:60-64`), has `supports_dense_mha_prefill = False` (`:38`), and reshapes rows into 64-row pages: "The SM120 kernels are built for 64-row pages only." (`:210-211`).
- The kernel call passes `sparse_mla_top_k=sparse_capacity`, `kv_scale_format=self.kv_scale_format` (`"arbitrary_fp32"` for model types starting with `glm`, else `"pow2_fp32"`), and no `backend=` argument (`:238-253, 27-31`).

**`backend="cake"` and `kv_cache_format="nvfp4"`:** neither is used on the SM12x MLA paths. `backend="cake"` occurs only in `vllm/model_executor/layers/mamba/ops/ssd_flashinfer.py:51` and the Kimi K3 drafter (`vllm/models/kimi_k3/nvidia/specialized_decode/drafter/attention.py:167, 655, 659`). `kv_cache_format=` is passed at `mla/flashinfer_mla_sparse.py:620` (the SM100 prims_ts path), `vllm/utils/flashinfer.py:546`, and a Kimi K3 file. The `nvfp4_ds_mla` dtype belongs to `FlashInferMLASparseTRTLLMBackend`, gated `capability.major == 10` (`mla/flashinfer_mla_sparse.py:99-106, 120-122`). [SRC]

**DeepSeek V4 / V4.1** are model-driven, not from the priority list: without an explicit backend, major 12 returns `DeepseekV4FlashInferSM120Attention`, other CUDA arches the FlashMLA class; naming `FLASHINFER_MLA_SPARSE` or `FLASHINFER_MLA_SPARSE_SM120` for a V4 model raises (`vllm/models/deepseek_v4/nvidia/model.py:556-579`; V4.1 equivalent at `vllm/models/deepseek_v41/nvidia/model.py:269-299`). The SM120 class additionally requires a FlashInfer dispatch-table entry for `(num_q_heads, top_k)`, read from the private `flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH` (`vllm/utils/flashinfer.py:554-567`; error text at `deepseek_v4/nvidia/flashinfer_sparse.py:657-667`). It calls the kernel with `kv_layout="NHD"` and `sinks=self.attn_sink` (`:851-864`).

**MLA prefill on non-SM100 GPUs:** `Platform.get_mla_prefill_backend_cls` gives major 10 a four-entry list and everything else `[MLAPrefillBackendEnum.FLASH_ATTN]` (`vllm/platforms/interface.py:472-492`). `FLASHINFER`, `TRTLLM_RAGGED`, and `TOKENSPEED_MLA` prefill backends each require `device_capability.major == 10` (`mla/prefill/flashinfer.py:58`, `trtllm_ragged.py:61`, `tokenspeed_mla.py:42`). On major 12 the FlashAttention version is 2, so `requires_v_padding` is true (V is zero-padded to the QK head dim; `mla/prefill/flash_attn.py:537-546`) and `flash_attn_supports_mla()` (FA3 on family 90 only, `fa_utils.py:250-270`) is false. [SRC + INF]

---

## 6. Composite and multimodal prefix

- `TRITON_FLASH_ATTN`: the causal child `_FABackend` requires `capability.major == 9` ("CUDA: Hopper only; Blackwell uses the Triton/FlashInfer composite") and FA3/FA4 (`triton_flash_attn.py:19-25, 55-61`).
- `TRITON_FLASHINFER`: created with `device_major=10`, `kernel_block_sizes=(64,)`, `head_sizes=(256, 512)` (`triton_flashinfer.py:12-22`); the factory enforces `capability.major == device_major` (`composite.py:411-415`).

Both composites are therefore invalid on major 12, and neither is in the major-12 priority list. [SRC]

**What a multimodal-prefix model gets on major 12 [INF from SRC]:**

1. `FLASH_ATTN` fails check 5, because `supports_mm_prefix()` needs FA4 (`flash_attn.py:418-420`), and `supports_combination` adds `"mm_prefix (PrefixLM bidirectional attention) requires FlashAttention v4, which does not resolve for this head_size"` (`:475-487`).
2. `FLASHINFER` is next. `_native_mm_prefix` (`flashinfer.py:126-149`) is true off SM100 when the model is `is_mm_prefix_lm`, DCP is 1, and the text config has no `attn_logit_softcapping`. `supports_combination` then requires the KV dtype to be in `(None, auto, float16, bfloat16, fp8, fp8_e4m3, nvfp4)` and no sinks (`:760-764`). The bidirectional ranges are served natively by `BatchPrefillWithCausalBidirectionalRangesWrapper` (`:1585-1603, 2613-2646`).
3. If FlashInfer is excluded (soft-capped models, head size outside `[64, 128, 256, 512]`, DCP), `TRITON_ATTN` is next: `supports_mm_prefix()` and `supports_sink()` both return `True` and capability is unrestricted (`triton_attn.py:383-388, 405-406`); `_cudagraph_support = ALWAYS` (`:97`).

Separately, the CUDA platform forces `disable_chunked_mm_input` for such models (`cuda.py:376-386`).

---

## 7. Other SM12x-relevant occurrences

Attention and KV cache:

| Location | Meaning |
|---|---|
| `v1/attention/backends/triton_attn.py:152-162` | 64 parallel softmax segments for small batches, keyed on **exact** `(12, 0)`; not applied on 12.1 |
| `v1/attention/backends/mla/indexer.py:321-324` | DSV4.1 indexer advertises only 64-token blocks on family 120 ("128 trips the block_kv assert, #59203") |
| `v1/attention/backends/mla/indexer.py:104-109` | `indexer_kv_dtype='mxfp4'` rejected off family 100, naming `sm_120` |
| `v1/attention/backends/mla/sparse_swa.py:373-376` | SM12x + Triton 3.6 illegal-memory-access workaround (clamped offset) |
| `v1/attention/backends/mla/sparse_swa.py:939-945` | FlashMLA scheduler metadata skipped on family 120 |
| `v1/attention/backends/mla/prefill/flash_attn.py:230-236` | FA4 MLA warmup keys returned for SM120 only when `num_splits == 1`; unreachable on 12.x since the version is not 4 [INF] |
| `model_executor/layers/attention/mla_attention.py:383-388, 720` | FP8 packing canonicalization and JIT warmup list for the SM120 sparse backend |
| `model_executor/warmup/flashinfer_sparse_mla_warmup.py:40-46, 133` | FlashInfer SM120 sparse-MLA autotune (label `DSv3.2`) |
| `models/deepseek_v4/sparse_mla.py:333-339` | adaptive-width decode disabled on major 12 (TODO) |
| `utils/deep_gemm.py:42-43` | sparse-indexer kernel block is 64 only on family 120 |
| `platforms/cuda.py:732-738` | `support_deep_gemm()` includes family 120 |
| `model_executor/layers/indexer_topk.py:186, 472` | a top-k path excluded on family 120 ("requires SM90+ and is not supported on the SM11x/SM12x families") |
| `models/minimax_m3/common/sparse_attention.py:578-600` | MiniMax M3 uses `flashinfer.msa_ops` on SM90 / SM120 / SM121 |
| `engine/arg_utils.py:3004-3006` | GB10 (12.1) kept on the small default `max_num_batched_tokens` tier |

Linear attention, GEMM, MoE (brief):

- GDN: on family 120 with `head_k_dim == 128` and CUDA runtime major >= 13, the FlashInfer path is enabled and the in-tree CuTe DSL kernel stays off (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:334-340`).
- Dense linear: FlashInfer cuDNN BF16 GEMM enabled only on family 120 (`model_executor/layers/utils.py:174-186`); b12x linear kernels each require family 120 (e.g. `model_executor/kernels/linear/scaled_mm/b12x.py:60-66`); config values `"b12x"` and `"flashinfer_b12x"` (`config/kernel.py:348-350, 430-431`).
- MoE: `flashinfer_b12x_moe.py:169`, `fused_moe/b12x.py:588`, DeepGEMM and CUTLASS MoE include family 120 (`deep_gemm_moe.py:475`, `cutlass_moe.py:729`); env `VLLM_B12X_MOE_FP4_FORCE_A16` (`vllm/envs.py:1801-1804`).
- Batch-invariant matmul tables contain an `"sm120"` entry selected by `capability.major == 12` (`model_executor/determinism/batch_invariant_configs.py:455, 1348-1349`).

**Docs.** `docs/design/attention_backends.md:180-187`:

> ### b12x
>
> The b12x backend supports causal decoder attention on NVIDIA SM120 and SM121 GPUs. b12x ships with FlashInfer (`flashinfer-python>=0.7.2`). Select it with:
>
> `vllm serve <model> --attention-backend b12x`

The tables are generated by `docs/mkdocs/gen_files/generate_attention_backends.py` (named at `attention_backends.md:3-6`, registered at `mkdocs.yaml:63`). The page's footnotes lag the code for SM12x: line 176 calls XQA "the SM90 decode path" and line 178 says "Default is FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise"; the generator's docstring likewise says "XQA on SM90" (`generate_attention_backends.py:1250`). The code enables XQA on family 120 and uses FA2 there. Lines 229-232 of the doc do state that `FLASHINFER_MLA_SPARSE_DSV4` is the SM12x default for DeepSeek V4.

---

## 8. Operator cheat-sheet for SM12x

Config key names are verified against `vllm/config/attention.py:43, 63, 80, 84, 87`.

| Goal | Flags | Expected log | Preconditions (source) |
|---|---|---|---|
| FA2 | `--attention-backend FLASH_ATTN` (optionally `--attention-config.flash_attn_version=2`) | `Using FlashAttention version 2` | no sinks, no FP8/NVFP4 KV, no mm-prefix, head size `% 8 == 0` and `<= 256` |
| FlashInfer + XQA decode | `--attention-backend FLASHINFER` (optionally `--attention-config.use_trtllm_attention=1`) | `decode_backend=xqa, ... arch=sm120` or `sm121` | cubins reachable; head dim in `[16, 256]`, multiple of 16; no DCP; `VLLM_BATCH_INVARIANT` unset |
| FlashInfer, XQA off | `--attention-backend FLASHINFER --attention-config.use_trtllm_attention=0` | `decode_backend=flashinfer-native` | sinks become unsupported; graph level drops to `UNIFORM_SINGLE_TOKEN_DECODE` |
| b12x | `--attention-backend b12x` | `Using b12x with q_heads=...` | bf16 model; head size 128/192/256; `--block-size` unset, 64, or 128; no DCP/PCP; no soft cap; `b12x.attention.paged` importable |
| Triton | `--attention-backend TRITON_ATTN` | `Using AttentionBackendEnum.TRITON_ATTN backend.` | none on capability |
| Sparse MLA | `--attention-backend FLASHINFER_MLA_SPARSE_SM120` | same pattern | `index_topk == 2048`, bf16 |
| Dense MLA | `--attention-backend TRITON_MLA` | same pattern | none |
| Per KV group | `--attention-config '{"backend_per_kind": {"sliding_window": "FLASHINFER"}}'` | — | keys are `KVCacheSpecKind` values (`config/attention.py:295-319`) |

Equivalent spellings: `--attention-config.backend X`, `-ac.backend X`, `-ac '{"backend": "X"}'`; do not combine with `--attention-backend`.

Relevant environment variables:

| Variable | Effect on SM12x | Source |
|---|---|---|
| `VLLM_BATCH_INVARIANT=1` | disables XQA (`supports_trtllm_attention` returns `False`); `FLASHINFER` and `B12X` fail validation; FA version forced away from 4; FA `num_splits=1`. The auto-default to `TRITON_ATTN` applies **only on XPU** | `utils/flashinfer.py:875-877`; `backend.py:221-223`; `fa_utils.py:164-169`; `flash_attn.py:846-847`; `arg_utils.py:2778-2787` |
| `VLLM_HAS_FLASHINFER_CUBIN=1` | asserts cubins are present, skipping the network probe that gates XQA | `utils/flashinfer.py:85-92`; `envs.py:2070-2071` |
| `FLASHINFER_CUBINS_REPOSITORY` | URL probed with a 5 s timeout | `utils/flashinfer.py:61-64, 851` |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | floor for the FlashInfer workspace (default 394 MiB); the actual size is `max(floor, max_num_batched_tokens * num_qo_heads * head_dim * 16)` | `envs.py:1974-1975`; `flashinfer.py:1381-1404` |
| `VLLM_KV_CACHE_LAYOUT` | overrides the KV layout (`NHD`, `HND`, `LBNHC`, ...) | `envs.py:270-273` |
| `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` | **not found** in this repository; if it exists it is consumed inside FlashInfer | whole-tree grep |

---

## 9. Assumptions, unresolved items, and falsifying experiments

**Assumptions the conclusions depend on:**

- The installed wheel matches this commit, and `vllm/vllm_flash_attn/_vllm_fa2_C` loads on the SM12x host. If FA2 fails to import, `get_flash_attn_version` returns `None` and section 2 changes.
- `get_device_capability()` reports `(12, 0)` or `(12, 1)`.
- For section 3.3, either the `flashinfer_cubin` package is installed or the NVIDIA artifactory URL is reachable from the serving host.

**Not determinable from this repository:**

1. Which kernel FlashInfer's `backend="auto"` resolves to on SM12x for prefill and for native decode.
2. Whether `flashinfer-python==0.7.4` contains `b12x.attention.paged`, and what `is_supported()` tests internally.
3. Everything inside `b12x.attention.paged` (`plan`, `Caps`, `*_graph_capacity`, `compile`, `run`): kernel structure, split policy, and actual scratch sizes.
4. Runtime cost of b12x plan construction and CuTe DSL compilation.
5. The forced-FA4 outcome in section 2.5 is a read-through, not an observed failure.
6. The docs generator's rendered tables were not built, so the published capability column for XQA was not checked.

**Smallest discriminating experiments:**

- *Default path:* on an SM12x host run `vllm serve <bf16 GQA model>` with no attention flags and grep for `Using FLASH_ATTN attention backend out of potential backends` and `Using FlashAttention version 2`. Any other backend name or version falsifies conclusion 1.
- *FP8 fallback:* add `--kv-cache-dtype fp8`; the selected backend should change to `FLASHINFER`, and the `FlashInfer resolved query dtypes` line should show `decode_backend=xqa` with both Q dtypes equal to the model dtype. `decode_backend=flashinfer-native` instead indicates the cubin probe failed, distinguishable by re-running with `VLLM_HAS_FLASHINFER_CUBIN=1`.
- *b12x availability:* `python -c "import importlib.util as u; print(u.find_spec('b12x'))"` in the serving environment. `None` means `--attention-backend b12x` will fail with `The b12x backend requires flashinfer-python>=0.7.2`.
- *Forced FA4:* `--attention-backend FLASH_ATTN --attention-config.flash_attn_version=4` with a head-size-128 model. Section 2.5 predicts the warning `FlashAttention 4 is not supported (FA4 is only supported on devices with compute capability 9.x, 10.x, or 11.x); it will not be used.` followed by `AssertionError: FlashAttention version not detected.`; a clean start on FA2 would falsify it.

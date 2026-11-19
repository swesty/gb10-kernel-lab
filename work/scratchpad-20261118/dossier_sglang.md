# Dossier: how SGLang serves attention on SM120 / SM121 (workstation Blackwell)

**Source tree:** SGLang `main` @ `b2cc023f23e14e6482490238968e018573847d0c` (commit date 2026-11-17), shallow clone, read-only. No repo code was executed. PR and issue text was read with `gh` on 2026-11-17.

**Citation form:** `sglang/<repo-relative path>:<line>`.

**Evidence labels used throughout:**
- **[SRC]** = read directly in the source tree at the commit above.
- **[PR] / [ISSUE]** = a claim made by a PR or issue author; not reproduced here.
- **[CALC]** = arithmetic done by hand from a formula quoted from source.
- **[INF]** = inference; the reasoning is stated next to it.

**Terms.** "SM12x" means compute-capability major 12 (both 12.0 and 12.1). "SM120" means exactly 12.0 (RTX PRO 6000 Blackwell, RTX 5090). "SM121" means exactly 12.1 (GB10 / DGX Spark). "FA4" means the CuTe-DSL FlashAttention-4 kernels under `python/sglang/kernels/ops/attention/`.

---

## 0. Verification of the facts handed in

| Handed-in claim | Verdict | Evidence |
|---|---|---|
| MHA default on SM12x is `flashinfer`, or `triton` when the model has attention sinks | Confirmed [SRC] | `sglang/python/sglang/srt/arg_groups/model_override_base.py:367-371` |
| MLA default on SM12x is `triton` | Confirmed [SRC] | `model_override_base.py:372-388` (SM12x misses `is_hopper_with_cuda_12_3`, `is_sm100`, `is_hip`, `is_mps`) |
| `fa3` raises on SM12x | Confirmed [SRC] | `sglang/python/sglang/srt/layers/attention/attention_registry.py:311-318` |
| `_load_fa4_kernels` sends major 12 to `flash_attention_v4_sm120` | Confirmed [SRC] | `sglang/python/sglang/srt/layers/attention/flashattention_backend.py:78-87` |
| `trtllm_mha` allowed on SM90/100/120 | Confirmed [SRC] | `sglang/python/sglang/srt/arg_groups/attention_hook.py:208-236` |
| pyproject pins `flashinfer_python[cu13]==0.7.4` | Confirmed [SRC] | `sglang/python/pyproject.toml:41` |

One stale comment to be aware of: the docstring of `get_default_attn_backend` still says "trtllm_mha does not support SM120, which will fall back to flashinfer" (`model_override_base.py:320-321`). The validation code in `attention_hook.py:208-236` contradicts the docstring: `trtllm_mha` is accepted on SM12x when requested explicitly. The docstring is correct only about the *default*.

---

## 1. SGLang's SM12x FA4 kernels

**Conclusion.** SGLang's SM12x FA4 path is an SGLang-owned CuTe-DSL kernel family in `fa4_sm120/` that plugs into the vendored upstream FA4 interface through a small dispatch bridge; the kernel uses warp-level `mma.sync` (16x8x16 F16/BF16) for both QK and PV, TMA for contiguous K/V, and `cp.async` gather with one dedicated loader warp for paged K/V.

### 1.1 Files (bytes) [SRC]

```
python/sglang/kernels/ops/attention/
  flash_attention_v4_sm120.py        10901   SGLang-facing API (varlen + kvcache + policy)
  flash_attention_v4.py              15216   generic FA4 API (helpers reused by the SM120 file)
  fa4_sm120/__init__.py                 73
  fa4_sm120/dispatch.py               2026   arch bridge (arch // 10 == 12)
  fa4_sm120/policy.py                 2919   low-head-dim decode tile qualification
  fa4_sm120/scheduler.py              5094   Sm120UniformBatchScheduler
  fa4_sm120/paged_kv.py               7970   Sm120PagedKVManager (cp.async page gather)
  fa4_sm120/flash_fwd.py            170860   FlashAttentionForwardSm120 (4554 lines)
  fa4_sm120/flash_fwd_decode.py      27106   FlashAttentionForwardSm120DecodeTranspose
  fa4_sm120/runtime.py               56229   Sm120ForwardPolicy / Sm120ForwardHost
  flash_attn/cute/interface.py      104104   vendored FA4 interface (calls the bridge)
  flash_attn/cute/{flash_fwd,flash_fwd_sm90,flash_fwd_sm100,flash_fwd_combine,pack_gqa,
                   paged_kv,pipeline,softmax,mask,tile_scheduler,...}.py   vendored FA4
  fp8_fa_sm120/{kernel,plan,fused_prep}.py   19705/6033/7687   (see 1.9)
  flash_mla_sm120.py 25751, flash_mla_sm120_triton.py 13324     (section 7)
```

What `flash_attention_v4_sm120.py` imports [SRC]: `fa4_sm120.dispatch` (`get_forward_arch`, `resolve_runtime_policy`, `try_cached_paged_decode`) and five helpers from the generic `flash_attention_v4.py` (`flash_attention_v4_sm120.py:12-23`).

### 1.2 Language and relationship to upstream

- **Language: CuTe DSL (Python, `cutlass.cute`)**, not CUDA C++ JIT [SRC]: `fa4_sm120/flash_fwd.py:10-26` imports `cutlass`, `cutlass.cute`, `cutlass.cute.nvgpu.{cpasync, warp}`, `cutlass.pipeline`, and `quack`.
- **Origin.** The only origin marker in the file is the upstream copyright line plus a one-line description [SRC] (`fa4_sm120/flash_fwd.py:1-2`):

```python
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
```

  `fa4_sm120/runtime.py:1` carries the same upstream copyright; `dispatch.py`, `policy.py`, `scheduler.py`, `flash_fwd_decode.py` and `flash_attention_v4_sm120.py` carry `Copyright (c) 2026, SGLang Team.` The package docstring says "SGLang-owned FlashAttention-4 kernels and launch policy for SM120" (`fa4_sm120/__init__.py:1`).
- **No file named `flash_fwd_sm120.py` exists** in the vendored `flash_attn/cute/` directory, and no string `sm120` appears in the vendored kernels other than `interface.py` [SRC, directory listing + grep].
- **[INF]** The SM120 kernel is a derivative work that started from Dao-AILab FA4 code (upstream copyright retained, subclass of the vendored `FlashAttentionForwardBase`, `flash_fwd.py:71`) and was then substantially rewritten by SGLang. A line-level diff against upstream `flash_fwd_sm120.py` could not be done: the upstream repository was not available in this environment. **Not determined:** which upstream commit it started from.
- **[PR #32991]** says the previous SM12x route was "an SM80-derived forward path" and that the PR "replace[s] the previous SM80-derived SM120 fallback".

Bridge code [SRC] (`fa4_sm120/dispatch.py:13-22`):

```python
@lru_cache(maxsize=None)
def get_forward_host(arch: int):
    """Return the optional architecture-owned forward host."""
    if arch // 10 == 12:
        from sglang.kernels.ops.attention.fa4_sm120.runtime import (
            sm120_forward_host,
        )

        return sm120_forward_host
    return None
```

`arch // 10 == 12` matches 120 and 121, so SM121 takes the same kernel family. The vendored interface accepts the arch only through the bridge (`flash_attn/cute/interface.py:669-672`).

### 1.3 MMA atom, warps, pipeline stages [SRC]

MMA atom for both GEMMs (`fa4_sm120/flash_fwd.py:970-991`, excerpt):

```python
        tiled_mma_qk = (
            cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                (1, num_qk_warps, 1),
                permutation_mnk=(self.tile_m, self.tile_n, 16),
            )
            if self.split_qk_n
            else cute.make_tiled_mma(
                warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
                (num_qk_warps, 1, 1),
                permutation_mnk=(num_qk_warps * 16, 16, 16),
            )
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
```

- Class docstring: "SM120 warp-MMA forward kernel with TMA fused into the QK warps." (`flash_fwd.py:72`).
- dtypes: only `Float16` and `BFloat16` (`flash_fwd.py:818-819`).
- **Pipeline depth is always 1** K stage: `get_fwd_num_stages` returns `1` unconditionally (`flash_fwd.py:691-696`). In the split-PV configurations there are additionally 2 P stages (`flash_fwd.py:785-786`) and 1 V stage (`flash_fwd.py:782-783`).
- **Threads** (`flash_fwd.py:698-734`): default 128 (4 warps); HD256 uses 64 (tile 16x64, 16x80), 128 (32x64), 192 (48x64), 256 (64x48, 64x64); low-head-dim paged decode uses `tile_m * 2`; HD256 paged decode tile 16x64 uses 128. The launch adds one extra loader warp when needed: `block=[self.num_threads + self._num_dma_threads(), 1, 1]` with `num_dma_threads = 32` (`flash_fwd.py:1348`, `:137`, `:797-802`).
- **Two warp organisations** (`flash_fwd.py:736-757`): for HD256 the kernel uses dedicated QK warps and dedicated PV warps that exchange P through shared memory (`_uses_split_pv_warps`); every other head dim has each warp do QK then PV on its own rows.
- Cluster shape is 1x1 (`flash_fwd.py:1611-1612`; `scheduler.py:41-44`), consistent with SM12x lacking multi-CTA `tcgen05` paths.

### 1.4 Tile sizes per head dim [SRC]

`get_fwd_tile_size` (`flash_fwd.py:399-689`) picks the first candidate that fits shared memory:

| (head_dim, head_dim_v) | Fallback candidate order (tile_m, tile_n) | Line |
|---|---|---|
| (64,64) or (128,128), paged decode, seqlen_q==1, visible K > 256 | (16 or 32, 64) | `policy.py:6-71`, `flash_fwd.py:462-463` |
| (256,256), packed Q rows <= 16 | (16,64), (16,80), (32,64), (64,64) | `flash_fwd.py:465-471` |
| (256,256), otherwise | (64,64), (64,48), (48,64), (32,64) (order changes in one row-count band) | `flash_fwd.py:472-492` |
| (32,32),(64,64),(96,96),(128,128),(192,128) "qualified" | (64, 128 for HD32 non-local else 64), then (64,64) | `flash_fwd.py:493-500` |
| anything else | (128,128) if max head dim <= 64, then (128,64), (64,64) | `flash_fwd.py:501-504` |

For batch 1 + packed GQA + causal/local, a preferred `tile_n` is chosen analytically from SM count (`_select_qualified_tile_n`, `flash_fwd.py:282-360`): for example HD64 gets N=128 when `seqlen_k < 1024`, otherwise N=192 if one wave fits else N=64 (`flash_fwd.py:345-348`); a "compact SM array" branch is `num_sms <= 64` (`flash_fwd.py:314`), which is the branch a 48-SM GB10 takes ([PR #32991] lists GB10 as 48 SMs).

### 1.5 Shared-memory budget vs 101376 bytes [SRC + CALC]

Budget formula (`flash_fwd.py:362-397`):

```python
        head_dim = math.ceil(head_dim / 16) * 16
        head_dim_v = math.ceil(head_dim_v / 16) * 16
        element_size = 2
        smem_usage_Q = tile_m * head_dim * element_size
        smem_usage_K = tile_n * head_dim * num_stages * element_size
        smem_usage_V = tile_n * head_dim_v * num_stages * element_size
        ...
        smem_usage = smem_usage_QV + smem_usage_K
        if (head_dim, head_dim_v, tile_m, tile_n) in ( (256, 256, 16, 64), ... (256, 256, 64, 64), ):
            smem_usage += 4 * tile_m * 4
        # Q/K/V/P/final full-empty mbarriers and conservative field alignment.
        return smem_usage + 2048
```

The capacity comes from `utils_basic.get_smem_capacity_in_bytes("sm_120")` (`flash_fwd.py:419`, `:842`); the literal 101376 does not appear in the kernel file. The same `"sm_120"` key is used on SM121 [INF: no `"sm_121"` lookup exists in `fa4_sm120/`].

[CALC] from the formula, 1 stage:

| (hd, hdv, tile_m, tile_n) | bytes | fits 101376? |
|---|---|---|
| (64,64,128,128) | 51200 | yes |
| (128,128,64,128) | 83968 | yes |
| (192,128,64,64) | 67584 | yes |
| (256,256,16,64) | 76032 | yes |
| (256,256,16,80) | 92416 | yes |
| (256,256,64,64) | **101376** | yes, exactly at the limit |
| (512,512,64,64) | 198656 | no |
| (512,512,16,64) | 149504 | no |

Consequences: HD256 at 64x64 consumes the entire 99 KB block, which is why HD256 is one-CTA-per-SM and single-stage (comment at `flash_fwd.py:716-717`). Head dim 512 (Gemma-4 full-attention layers) has no fitting candidate and hits `ValueError: ... exceeds SM120 shared-memory capacity` (`flash_fwd.py:686-689`); [PR #45871] independently states "BF16 HD512 previously raised `exceeds SM120 shared-memory capacity`".

### 1.6 K/V staging: TMA vs cp.async, and page size [SRC]

- **Contiguous K/V:** TMA (`cpasync.CopyBulkTensorTileG2SOp`, `PipelineTmaAsync`), `flash_fwd.py:1193-1223`, `:1456-1472`.
- **Paged K/V:** `cp.async` gather through `Sm120PagedKVManager` with a 32-thread loader group and `PipelineCpAsync`, `flash_fwd.py:1473-1504`; the copy atom is `cpasync.CopyG2SOp(cache_mode=...GLOBAL)` with 128-bit copies (`paged_kv.py:58-74`). The call asserts `mPageTable is None or self.paged_kv`, "SM120 paged KV requires the dedicated DMA-warp specialization" (`flash_fwd.py:1093-1095`).
- **Page table handling:** each loader thread does a fast divmod of the row index by page size, reads `mPageTable[page_idx]`, and builds a raw global pointer into a `(page_size, head_dim, num_pages)` view (`paged_kv.py:113-147`). Any page size works on the gather path, including page size 1.
- **Paged TMA exists only in one specialization.** `paged_tma = False` on the base kernel with the comment "Experimental same-page paged-KV TMA path... qualified dispatch keeps it disabled" (`flash_fwd.py:74-76`); `paged_tma = True` on `FlashAttentionForwardSm120DecodeTranspose` (`flash_fwd_decode.py:41`). That specialization is selected only when `page_size == tile_n` (64), HD256, single-token decode, packed GQA with 2 to 8 packed rows, at most 8 batch-times-KV-head groups, and dense page strides (`runtime.py:919-932`).

**Page-size-128 forcing: confirmed gated on `is_sm100` only** [SRC] (`sglang/python/sglang/srt/arg_groups/overrides.py:1486-1510`):

```python
def _fa4_page_constraint(view: Any) -> dict:
    if (
        (
            view.attention_backend == "fa4"
            or view.decode_attention_backend == "fa4"
            or view.prefill_attention_backend == "fa4"
        )
        and not use_mla_backend(view)
        and get_platform().is_sm100
        and (view.speculative_eagle_topk or 0) <= 1
    ):
        ...
        return {"page_size": 128}
    return {}
```

On SM12x `get_platform().is_sm100` is false (major 10 only, `sglang/python/sglang/srt/utils/device.py:318-322`), so the pass returns `{}` and page size stays at the operator's value or the default of 1 (`overrides.py:1610-1632`). What SM12x then does: page size 1 (default) runs the cp.async gather kernel; `--page-size 64` additionally unlocks the page-TMA transposed HD256 decode kernel. **[INF]** `--page-size 64` is the tuned decode setting, because [PR #32991] benchmarked paged decode at "page size 64" and `runtime.py:923` requires `page_size == tile_n`. The docs table still lists FA4 native page size as "128" with no SM12x qualifier (section 9).

### 1.7 Split-KV (`num_splits`) [SRC]

Launch policy (`dispatch.py:25-38`):

```python
    arch = device_capability[0] * 10 + device_capability[1]
    uses_arch_decode_policy = get_forward_host(arch) is not None
    no_splitkv = device_capability < (9, 0) or uses_arch_decode_policy
    num_splits = 1 if deterministic or no_splitkv else 0
    decode_num_splits = (
        0 if uses_arch_decode_policy and not deterministic else num_splits
    )
    return num_splits, decode_num_splits, uses_arch_decode_policy
```

On SM12x: **prefill `num_splits = 1` (no split-KV)**; **decode `num_splits = 0` (auto)** unless `--enable-deterministic-inference`, in which case both are 1. The third value sets `_decode_uses_static_max_seqlen_k` in the backend (`flashattention_backend.py:375-384`), which makes CUDA-graph capture use `max_context_len` as the K bound (`flashattention_backend.py:793`, `:2403-2408`). The backend passes `decode_num_splits` only for plain decode; sliding-window layers, cascade (topk>1) and prefill-aware SWA use `num_splits` = 1 (`flashattention_backend.py:2415-2421`).

Auto heuristic (`runtime.py:670-749`), three branches:
1. **Low-HD (64/128) paged decode with tile_m 16:** `_select_low_hd_decode_num_splits` (`runtime.py:305-368`): no split at <= 4 KV tiles; cap 8 splits (`policy.py:10`); integrated GPUs with few CTAs target two thirds of the SM array, capped at 32.
2. **HD256 paged decode:** `_select_decode_num_splits` (`runtime.py:210-302`): no split at <= 4 KV tiles; candidates are powers of two up to 128 plus the one-wave maximum; a fitted latency/bandwidth model (`_predict_decode_split_us`, `runtime.py:157-207`) picks the minimum. The constants are described as "Joint fit of interleaved NCU duration curves from the 110-SM/384-bit and 188-SM/512-bit SM120 SKUs" (`runtime.py:45-48`).
3. **Otherwise:** the generic upstream heuristic with max 128.

**SM121-specific behaviour inside the heuristic** [SRC]: the hardware probe reads `is_integrated`, `memory_bus_width` and `memory_clock_rate` (`runtime.py:125-146`) and has dedicated integrated-GPU branches, e.g. "Three unsplit CTAs per LPDDR channel already saturate the LPDDR path" returns 1 split (`runtime.py:245-248`). **[INF]** GB10 is the integrated device these branches target; the fitted constants themselves were calibrated on the two discrete SKUs only, per the comment quoted above.

### 1.8 Feature handling [SRC unless noted]

| Feature | SM12x FA4 behaviour | Evidence |
|---|---|---|
| GQA packing | `pack_gqa` defaults to true when q heads / kv heads > 1; kernel reshapes Q, O, LSE with `pack_gqa_layout` | `interface.py:686-688`; `flash_fwd.py:1180-1191` |
| Attention sinks | supported: `sinks` forwarded as `learnable_sink`; must be bf16, shape `(num_head,)` | `flash_attention_v4_sm120.py:118-119`; `interface.py:646-648`; kernel arg `flash_fwd.py:1084` |
| Sliding window | supported: `is_local` plus window sizes are kernel args; decode visible span computed exactly | `flash_fwd.py:1082-1083`; `policy.py:15-27`; `runtime.py:751-778` |
| Block sparsity | not supported: `assert blocksparse_tensors is None` | `flash_fwd.py:1091` |
| Relative bias | supported for head dim <= 128 with tile (64,128) | `runtime.py:626-634` |
| FP8 KV (e4m3/e5m2) | **not supported at this commit**: `assert arch // 10 == 10, "FP8 is only supported on SM100..."`; the SM90 in-kernel cast path is gated `arch // 10 == 9` | `interface.py:787-790`, `:598-604` |
| MXFP8 KV | arg-level check passes on SM12x (`is_blackwell` is major 10/11/12) | `kv_cache_hook.py:103-112`. **[INF]** it then fails in the kernel interface, because block-scaled inputs are fp8 and reach the `arch // 10 == 10` assert. Not verified by execution. |
| NVFP4 KV (`--kv-cache-dtype nvfp4`) | FA4 is excluded: on non-SM100 the resolver forces prefill `flashinfer` (FP8 E4M3 dequant workspace) and decode `trtllm_mha` (XQA); an explicit `--prefill-attention-backend fa4` raises | `kv_cache_hook.py:28-31`, `:61-93` |
| Other FP4 KV recipe (`fp4_mx_block16`), "KV4" | with FA4 prefill on an MHA model the decode backend must be `triton`, `torch_native` or `flex_attention`. **[INF]** this branch is reached for the non-NVFP4 recipe; the enclosing condition (`kv_cache_hook.py:120-160`) was not read in full | `kv_cache_hook.py:318-337` |
| MLA through FA4 (`qv`) | rejected: `assert arch // 10 in [10, 11], "only support Blackwell arch with qv"` | `interface.py:1084-1085` |
| Spec decoding topk>1 | FA backend runs the two-pass cascade with `num_splits` = 1; at `page_size > 1` FA4 is rejected for EAGLE (allowed list is `flashinfer`, `fa3`, `triton`) and for NGRAM (flashinfer only) | `flashattention_backend.py:2252`, `:2415-2421`; `speculative_hook.py:1464-1475`, `:1548-1559` |
| Hybrid GDN models | FA4 not in the SM12x allowed set `{"triton", "trtllm_mha", "flashinfer"}` | `attention_registry.py:536-550` |
| Head dim 512 | raises (section 1.5) | `flash_fwd.py:686-689` |

### 1.9 `fp8_fa_sm120/`

`fp8_fa_sm120/` is used by the diffusion stack (`multimodal_gen/runtime/layers/attention/backends/fp8_fa_sm120_attn.py`), not by the LLM `FlashAttentionBackend` [SRC, grep of importers]. Its internals were not analysed.

---

## 2. Pull requests

### PR #32991 "feat(attention): add architecture-owned SM12x FA4 kernels"
- **State:** MERGED 2026-08-13; +10449 / -198; author `lucifer1004`. `reviewDecision` reports `REVIEW_REQUIRED` although merged.
- **Body [PR]:** SM12x FA4 previously ran "an SM80-derived forward path" without TMA dataflow, paged KV, SplitKV or relative bias. Adds "TMA-based SM12x forward and paged-decode kernels with warp MMA", SM-count-aware tile selection, BF16/FP16, MHA/GQA/MQA, varlen, paged KV, SplitKV, sinks, relative bias, launch-plan caching, CUDA-graph-safe decode plans.
- **Tests [PR]:** `test_flash_attention_4_sm120.py: 53 passed`. Coverage listed: dense/varlen, BF16/FP16, MHA/GQA/MQA, asymmetric Q/V dims, packed/unpacked GQA, paged KV, SplitKV, sinks, relative bias, preallocated output, CUDA Graph replay, PDL, specialization-cache ordering. CI comments show the file passing on a `1-gpu-5090` runner three times on 2026-08-12. [SRC] the test file at this commit has 20 `def test_` functions; **[INF]** 53 comes from parametrization (not confirmed).
- **Accuracy [PR]:** 684 prefill cases vs Triton; max abs diff 0.0078125 (RTX PRO 6000, RTX PRO 5000), 0.015625 (GB10).
- **Prefill speed [PR]** (228 cases per GPU):

| GPU | SMs | PR / Triton | PR / SM80 fallback |
|---|---|---|---|
| RTX PRO 6000 Blackwell | 188 | 3.027x (228/228 wins) | 2.377x (222/228) |
| RTX PRO 5000 Blackwell | 110 | 3.311x (228/228) | 2.104x (224/228) |
| GB10 (SM121) | 48 | 3.610x (228/228) | 1.730x (205/228) |

  Absolute examples [PR]: GB10, MHA BF16 HD256 L6144: PR 2108.42 us vs Triton 13977.18 us vs SM80 fallback 1823.94 us (the PR loses to the old fallback in this case). RTX PRO 6000 same case: 456.29 / 2588.77 / 571.95 us.
- **Paged decode [PR]** (108 cases per GPU, page size 64, CUDA-graph replay): 1.311x (RTX PRO 6000, 92/108 wins), 1.317x (RTX PRO 5000, 96/108), 1.214x (GB10, 90/108) over Triton.
- A later comment (2026-11-07) reports a bug in the kernel with a fix in PR #49044; PR #49044 was not read.

### PR #45871 "[Perf] [NVIDIA] attention: Native FP8 KV cache for SM120 FA4 kernels"
- **State:** **OPEN, not merged**; `REVIEW_REQUIRED`; zero comments; +649 / -68; author `viethq18`; created 2026-10-22. CI slots all red at time of reading.
- **Motivation [PR]:** on RTX PRO 6000, `--attention-backend fa4 --kv-cache-dtype fp8_e4m3` fails with `AssertionError: FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe.` This matches [SRC] `interface.py:787-790` at the dossier commit.
- **Design [PR]:** QK and PV on the E4M3 `m16n8k32` warp MMA; V kept as E4M3 in smem; P pre-scaled by 256 before the E4M3 cast; per-(batch, kv_head) descales; BF16 output; new tile candidates for head dim > 256 that fit "the 101376 B SM120 budget".
- **Numbers [PR]** (1x RTX PRO 6000): kernel microbench, 9216 KV tokens: decode HD256 213.6 -> 105.0 us (2.03x); decode HD512 1018.3 -> 549.6 us (1.85x); prefill HD256 175.5 -> 98.5 us (1.78x); prefill HD512 12871.6 -> 5880.1 us (2.19x). Serving `gemma-4-31B-it`, ISL 8k / OSL 1k: BF16 max concurrency 20 at 383.36 tok/s vs FP8 46 at 863.56 tok/s (2.25x). GSM8K-Platinum accuracy 0.730 (BF16) vs 0.742 (FP8). Tests: "62 passed (9 new) on SM120". No SM121 numbers.

---

## 3. `trtllm_mha` on SM12x

**Conclusion.** On SM12x `trtllm_mha` uses FlashInfer's FMHAv2 kernel for prefill and FlashInfer's XQA implementation for decode, both selected by a major-12 predicate, so SM121 is routed identically to SM120.

Module header [SRC] (`sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py:6-12`):

```
Prefill dispatch:
  - SM90 / SM120: trtllm_fmha_v2_prefill (Q_PAGED_KV_NHD layout)
  - SM100:        trtllm_batch_context_with_kv_cache (HND layout)

Decode: uses XQA on SM90 and SM120, TRTLLM-GEN on SM100.

Sliding window and attention sink features are supported.
```

- Gates: `self.is_xqa_impl = get_platform().is_sm90 or get_platform().is_sm120` (`:309`) and `self.use_fmha_v2 = get_platform().is_sm90 or get_platform().is_sm120` (`:417`). `get_platform().is_sm120` resolves to `is_sm120_supported`, which is major 12 (`sglang/python/sglang/srt/runtime_context.py:1855`).
- **Prefill call:** `flashinfer.prefill.trtllm_fmha_v2_prefill(qkv, input_layout="Q_PAGED_KV_HND"|"Q_PAGED_KV_NHD", ..., block_tables=..., mask_mode="causal"|"padding", window_left=..., sinks=..., skip_softmax_threshold_scale_factor=...)` (`:2046-2070`). FMHAv2 derives strides from shapes, so SGLang copies the batch's pages into a `[pages, 2, page_size, heads, head_dim]` tensor first (`:1565-1582`).
- **Decode call:** `flashinfer.decode.trtllm_batch_decode_with_kv_cache(query=..., kv_cache=..., block_tables=..., window_left=..., sinks=..., ...)` (`:1643-1660`). SGLang passes no `backend=` argument. **[INF]** FlashInfer picks XQA internally from the device arch; the FlashInfer source was not in the tree to confirm.
- XQA differences coded in SGLang: Q stays BF16 even with FP8 KV (`:1778-1789`); `SGLANG_TRTLLM_MHA_FULL_ATTN_KV_SPLITS` is forced to 0 because "the XQA decode kernels pick their own split" (`:466-469`); compact ragged verify is rejected (`:1301-1307`); native NVFP4 prefill is rejected (`:310-315`).

**Restrictions** [SRC] (`attention_hook.py:216-230`, verbatim):

```python
        if (
            prefill_backend == "trtllm_mha"
            and get_platform().is_sm120
            and (
                cfg.kv_cache_dtype == "fp8_e4m3"
                or (
                    envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get() or 0.0
                )
                > 0
            )
        ):
            raise ValueError(
                "TRTLLM FMHAv2 prefill on SM120 does not support "
                "fp8_e4m3 KV cache or skip-softmax."
            )
```

- The restriction is on **prefill only**; `--decode-attention-backend trtllm_mha` with FP8 KV is not rejected by this check.
- Context parallelism with `trtllm_mha` prefill requires SM100 (`attention_hook.py:237-246`).
- **Page sizes:** code accepts 16, 32, 64 or 128 and snaps anything else to 64 (`overrides.py:1387-1400`). The docs table says "16, 32 or 64" and "XQA backend only works well for pagesize 64" (section 9). The GQA/equal-head-dim requirement for page 128 is skipped on XQA: "XQA (SM90/SM120 decode) has native page-128 kernels; no check needed" (`trtllm_mha_backend.py:423-424`). The default page size of 1 is therefore snapped to 64 with a warning.
- **Spec decoding topk>1:** rejected: `"trtllm_mha backend only supports topk = 1 for speculative decoding."` (`sglang/python/sglang/srt/arg_groups/speculative_hook.py:1369-1373`). For topk=1 verify on XQA a precomputed causal bit mask is passed (`trtllm_mha_backend.py:480-491`, `:2019-2023`).
- **Hybrid GDN models:** on SM12x both phases must be in `{"triton", "trtllm_mha", "flashinfer"}` (`attention_registry.py:536-550`). The docs are stricter than the code: docs say `trtllm_mha` is "supported for `--decode-attention-backend` only" on SM120, but the assertion allows it for prefill too. The mamba `no_buffer` strategy rejects `trtllm_mha` (`sglang/python/sglang/srt/arg_groups/mamba_hook.py:166-167`).

---

## 4. `flashinfer` backend on SM12x

**Conclusion.** On SM12x the `flashinfer` backend runs FlashInfer's generic `fa2` kernels for paged prefill and decode, and lets FlashInfer choose (`"auto"`) for ragged prefill; the only arch branch in the file is for SM100.

[SRC] `sglang/python/sglang/srt/layers/attention/flashinfer_backend.py`:
- `self.prefill_backend = "fa2"`, `self.decode_backend = "fa2"` (`:373-374`).
- Arch branch (`:550-571`): `fmha_backend = "auto"`; only `if get_platform().is_sm100:` changes it to `"fa2"` or `"cutlass"`. SM12x keeps `"auto"`.
- Wrappers (`:578-616`): `BatchPrefillWithRaggedKVCacheWrapper(..., "NHD", backend=fmha_backend)`, two `BatchPrefillWithPagedKVCacheWrapper(..., "NHD", backend="fa2")` (paged and verify), `BatchDecodeWithPagedKVCacheWrapper(..., "NHD", backend="fa2", use_tensor_cores=...)`.
- **[INF]** What `"auto"` resolves to on SM12x inside FlashInfer 0.7.4 was not determined (FlashInfer source not in tree).
- **Why sinks force `triton`:** the default selector says so directly (`model_override_base.py:368-371`):

```python
            # FlashInfer does not support attention sinks.
            if get_platform().has_flashinfer and not model_config.has_attention_sinks:
                return "flashinfer"
            return "triton"
```

  Consistent with that, `flashinfer_backend.py` contains no `sinks` argument anywhere [SRC, grep].
- Triton on SM12x uses smaller tiles because of the smem limit (`sglang/python/sglang/kernels/ops/attention/extend_attention.py:101-108`): `(64,128)` for head dim <= 128, `(64,64)` for <= 256, else `(32,32)`.

---

## 5. Arch helpers and the "major == 12" vs "== 12.0" split

### 5.1 Helpers [SRC] (`sglang/python/sglang/srt/utils/device.py`)

| Helper | Lines | Exact semantics |
|---|---|---|
| `_check_cuda_device_version(majors, cuda_version)` | 281-291 | `get_device_sm() // 10 in majors` and torch CUDA version >= `cuda_version` |
| `is_blackwell_supported()` / `is_blackwell` | 301-308 | majors `[10, 11, 12]`, CUDA >= 12.8. **True on SM120 and SM121.** |
| `is_sm120_supported()` | 311-315 | majors `[12]`, CUDA >= 12.8. **True on SM120 and SM121** despite the name. |
| `is_sm100_supported()` | 318-322 | majors `[10]` only |
| `is_sm100_or_sm110_supported()` | 325-331 | majors `[10, 11]`; comment "excludes consumer Blackwell (SM120)" |
| `is_sm120()` | 355-358 | `torch.cuda.get_device_capability() == (12, 0)`; comment "this excludes SM121/GB10" |
| `is_sm121()` | 361-365 | `== (12, 1)`; comment "GB10 (DGX Spark and OEM equivalents)" |

`get_platform().is_sm120` maps to **`is_sm120_supported`** (major 12), and `get_platform().is_blackwell` to `is_blackwell_supported` (`runtime_context.py:1855-1856`). So `get_platform().is_sm120` and `device.is_sm120()` have **different meanings** on a DGX Spark: the first is true, the second false.

Kernel-side arch object [SRC] (`sglang/python/sglang/kernels/arch/_cuda.py:26-44`): `sm_version = major*10 + minor`; `is_sm(major, minor=None)` matches any minor when `minor` is omitted; `is_sm_at_least(major, minor)`; `supports_pdl = major >= 9`. JIT compile suffix (`sglang/python/sglang/kernels/jit/utils/arch.py:83-102`): SM12.0 compiles as `sm_120f` when nvcc >= 12.9, SM12.1 (and SM12.0 with older nvcc) as `...a`, because "SM120/SM121 need separate cubins to avoid cudaErrorIllegalInstruction".

### 5.2 Call sites where the exact-minor choice changes DGX Spark behaviour [SRC]

Exact `(12, 0)` — **DGX Spark excluded**:

| File:line | Meaning for SM121 |
|---|---|
| `srt/layers/attention/qwen_sparse_attn_backend.py:108` | QSA sparse decode via `trtllm_batch_decode_with_kv_cache` disabled; comment: "it silently corrupts long-context decode on SM121/GB10" |
| `srt/layers/moe/moe_runner/flashinfer_cutlass.py:284`, `:323` | fused-finalize FP4 MoE fast path is SM120-only |
| `srt/models/nemotron_h.py:373`, `:853` | Nemotron-H prefill-overlap path SM120-only |
| `kernels/ops/gemm/sm120_bf16.py:59-60`, `:105` | SM120 BF16 GEMM kernel not used on Spark |
| `kernels/kda_kernels/sm120_fp8.py:129`; `kernels/kda_kernels/qwen3x_nvfp4_gemm.py:33` | SM120-only FP8 / NVFP4 GEMM kernels |
| `kernels/ops/layernorm/mhc.py:2368` | one MHC fast path SM120-only |
| `srt/model_loader/utils.py:285` (`get_device_sm() == 120`) | dense block-FP8 GEMM diverted away from DeepGEMM on SM120 only; SM121 keeps the DeepGEMM route |
| `kernels/jit/utils/arch.py:97` | `f` vs `a` compile target |

Exact `(12, 1)` — **DGX Spark only**:

| File:line | Meaning |
|---|---|
| `srt/layers/attention/qwen_sparse_attn_backend.py:121-124` | QSA varlen attention uses `qwen38_qsa_sm121_varlen` instead of FA4/FA2 |
| `kernels/kda_kernels/qwen38_qsa_sm121/__init__.py:55` | guard of that SM121 kernel |
| `kernels/ops/moe/inkling_moe.py:973` | `num_stages` 3 instead of 4: "sm_121 ... caps shared memory at 99 KB per block and num_stages=4 needs 108 KB" |

Major 12 — **SM120 and SM121 treated alike** (attention-relevant subset):

| File:line | Meaning |
|---|---|
| `srt/layers/attention/flashattention_backend.py:80` | FA4 kernel family selection |
| `kernels/ops/attention/fa4_sm120/dispatch.py:16`, `runtime.py:410` | FA4 host bridge |
| `srt/layers/attention/trtllm_mha_backend.py:309`, `:417` | XQA decode, FMHAv2 prefill |
| `srt/arg_groups/attention_hook.py:210`, `:218`, `:232` | `trtllm_mha` allow + FP8/skip-softmax rejection |
| `srt/layers/attention/attention_registry.py:537` | hybrid GDN allowed set |
| `srt/arg_groups/overrides.py:690` | GLM DSA + FP8 KV defaults to `flashinfer_sparse_mla` |
| `kernels/ops/attention/flash_mla_sm120.py:690` | `flashinfer_sparse_mla` validity |
| `srt/arg_groups/overrides.py:1313` | deterministic-inference default backend |
| `srt/layers/attention/deepseek_v4_backend.py:262`, `:1679`, `:3097`, `:4578`; `srt/models/deepseek_v4.py:1235`; `srt/layers/attention/dsv4/metadata.py:29`; `dsv4/indexer.py:869` | DSV4 SM12x paths (section 7) |
| `srt/layers/deep_gemm_wrapper/configurer.py:44`, `:59` (`in (120, 121)`) | DeepGEMM SM12x API probe and UE8M0 scales |
| `kernels/ops/attention/extend_attention.py:101` | Triton tile sizes |
| `srt/arg_groups/kv_cache_hook.py:168` | NVFP4 KV allowed |

Negative gates that exclude both (`is_sm100` only): FA4 page-128 forcing (`overrides.py:1495`), `cutedsl_mla` (`overrides.py:1457`), default `trtllm_mha` (`model_override_base.py:350`), `DEEPGEMM_BLACKWELL` (`configurer.py:57`), FA4 `qv` and FP8 (`interface.py:1085`, `:789`).

---

## 6. Per-phase flags

**Conclusion.** Three flags share one choice list; a per-phase flag overrides `--attention-backend` for its phase; differing phases get a hybrid wrapper; on SM12x the valid dense-MHA backends are `flashinfer`, `triton`, `fa4`, `trtllm_mha`, `torch_native`, in any prefill/decode pairing except those rejected below.

- **Definition** [SRC]: `sglang/python/sglang/srt/arg_groups/fields/exec_.py:237-245` (`attention_backend`), `:246-254` (`decode_attention_backend`, "have priority over --attention-backend"), `:267-275` (`prefill_attention_backend`). All use `choices=ATTENTION_BACKEND_CHOICES`.
- **Allowed values** [SRC] (`sglang/python/sglang/srt/arg_groups/choices.py:81-112`): `triton, torch_native, flex_attention, dsa, nsa, qsa, unlimited_ocr_v2, dsv4, ernie55, fa3, fa4, flashinfer, llada2_cfg_flashinfer, flashmla, trtllm_mla, cutedsl_mla, tokenspeed_mla, trtllm_mha, hpc_ops, minicpm_flashattn, minicpm_flashinfer, aiter, wave, intel_amx, ascend, intel_xpu`.
- **Resolution** [SRC]: `attention_backends_of` falls back per phase to the base backend (`model_override_base.py:154-168`). `_attention_backend_default` sets the base to the prefill value when prefill == decode, else fills the default when the base is unset (`overrides.py:1339-1353`). Docs: differing phases enable `hybrid_attn_backend.py`.
- **Validation that rejects combinations on SM12x** [SRC]:

| Combination | Result | Evidence |
|---|---|---|
| `fa3` in either phase | AssertionError (needs major 8 or 9) | `attention_registry.py:311-318` |
| `trtllm_mha` prefill + `--kv-cache-dtype fp8_e4m3` or skip-softmax | ValueError | `attention_hook.py:216-230` |
| `trtllm_mha` prefill + `attn_cp_size > 1` | ValueError | `attention_hook.py:237-246` |
| any `trtllm_mha` + `speculative_eagle_topk > 1` | ValueError | `speculative_hook.py:1369-1373` |
| `fa4` (base) + topk > 1 + `page_size > 1` | ValueError | `speculative_hook.py:1464-1475` |
| `fa4` + FP8 KV | AssertionError in kernel interface at first call | `interface.py:787-790` |
| `--kv-cache-dtype nvfp4` + explicit prefill backend other than `flashinfer` (SM12x cannot use native NVFP4 prefill: "currently requires SM100") | ValueError | `kv_cache_hook.py:61-79` |
| `--kv-cache-dtype nvfp4` + explicit decode backend other than `trtllm_mha` | ValueError | `kv_cache_hook.py:81-87` |
| FP4 KV ("KV4") + `fa4` prefill on MHA + decode not in `triton/torch_native/flex_attention` | AssertionError | `kv_cache_hook.py:328-337` |
| hybrid GDN model + `fa4` | AssertionError | `attention_registry.py:536-550` |
| `cutedsl_mla` | ValueError (SM100 only) | `overrides.py:1457-1460` |
| deterministic inference + `trtllm_mha` | ValueError (not in `ascend, fa3, fa4, flashinfer, intel_xpu, triton`) | `choices.py:128-135`, `overrides.py:1288-1306` |

- **`fa4` prefill + `trtllm_mha` decode on SM12x:** no validation rejects it for a plain MHA model with a BF16 KV cache [SRC: none of the checks above fire]. `trtllm_mha` snaps page size to 64 (`overrides.py:1396-1400`), which FA4's gather kernel accepts. **[INF]** valid at the argument layer; runtime compatibility of the shared KV layout was not verified by execution. The docs note "trtllm_mha being incompatible with FA4" applies to the KV4 (FP4 KV) scenario.

---

## 7. MLA and sparse attention on SM12x

- **`trtllm_mla` gate** [SRC] (`overrides.py:1417-1431`): requires `get_platform().is_blackwell` (majors 10/11/12), message "only supported on Blackwell GPUs (SM100/SM12x)"; KV dtype in `fp8_e4m3, bf16, auto`; page size snapped to 32 or 64 (`overrides.py:1368-1376`). So the argument layer **admits SM12x**. **[ISSUE #40286]** reports that on SM121 the related sparse call `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla` fails with `TllmGenFmhaRunner ... Unsupported architecture`. **[INF]** dense `trtllm_mla` on SM12x likely fails the same way at runtime; not verified.
- **MLA default on SM12x is `triton`** (`model_override_base.py:372-388`); FA4 MLA is unavailable (`interface.py:1085`).
- **DSV4 (`DeepseekV4ForCausalLM`)** [SRC]: override forces `attention_backend = "dsv4"` and `page_size = 256` (`sglang/python/sglang/srt/arg_groups/model_overrides/deepseek_v4.py:18`, `:29`, `:59-69`). Backend class is chosen by `--dsv4-attn-backend` (`deepseek_v4_trtllm_backend.py:551-561`). On SM12x the sparse attention call goes to `flash_mla_with_kvcache_sm120` (`deepseek_v4_backend.py:4578-4582`), FlashMLA metadata is skipped (`:261-263`), and one sparse-prefill fast path is disabled (`:3097`).
- **`SGLANG_SM120_FLASHMLA_BACKEND`** [SRC] (`sglang/python/sglang/srt/environ.py:1447-1448`):

```python
    # SM120 FlashMLA decode backend: "flashinfer" (default), "triton", or "torch".
    SGLANG_SM120_FLASHMLA_BACKEND = EnvStr("flashinfer")
```

  Dispatch (`kernels/ops/attention/flash_mla_sm120.py:298-374`): `"flashinfer"` -> for more than 64 query tokens (`SM120_DECODE_MAX_TOKENS = 64`, `:212`) `flashinfer.mla._sparse_mla_sm120._sparse_mla_sm120_paged_attention` (`:227`, `:252`), otherwise `_flash_mla_flashinfer`; `"triton"` -> `flash_mla_sparse_decode_triton`; **any other string** falls through to the pure-PyTorch `_sm120_sparse_decode_fwd` (`:362-374`). The variable is an unvalidated `EnvStr`, so a typo silently selects the PyTorch path. SGLang's 256-token pages are split into four 64-token pages for FlashInfer (`:377-391`, `:597-600`); `SGLANG_OPT_SM120_DIRECT_SWA_KV` stores that layout directly (`environ.py:1449-1451`). Note: the module docstring still names an older variable `SGLANG_SM120_TRITON_FLASHMLA` (`flash_mla_sm120.py:6-7`) that is not what the code reads.
- **DSA (DeepSeek V3.2 / GLM) defaults on SM12x** [SRC] (`overrides.py:756-773`): FP8 KV -> `trtllm` for both phases (`major >= 10`); BF16 -> prefill `flashmla_sparse`, decode `trtllm`. Exception: `GlmMoeDsaForCausalLM` + major 12 + FP8 KV -> `flashinfer_sparse_mla` for both (`overrides.py:688-722`), and that backend is valid only for `GlmMoeDsaForCausalLM` / `GlmMoeDsaForCausalLMNextN` (`flash_mla_sm120.py:29-32`, `:686-701`). The `triton` DSA backend is ROCm-only (`overrides.py:599-605`). `triton_sparse_mla` prefill is capped at 32 query heads, with the comment that more "makes the [BLOCK_H, 512] fp32 accumulator spill or miss SM120's smem budget" (`srt/layers/attention/dsa_backend.py:500-501`).
- **DeepGEMM on SM12x** [SRC] (`sglang/python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:41-60`): enabled on 120/121 only if the installed build exposes `fp8_einsum`, `m_grouped_fp8_fp4_gemm_nt_contiguous`, `transform_sf_into_required_layout` ("mma.sync block-scale, no TMEM ... DeepGEMM#324"). `DEEPGEMM_BLACKWELL` is SM100-only, so `DEEPGEMM_PAGED_SPARSE_MQA_LOGITS` is always false on SM12x (`:57`, `:64-78`). Attention indexers: the dense `fp8_fp4_mqa_logits` path is used on major >= 10 (`srt/layers/attention/dsv4/v41_indexer/__init__.py:60-81`); the sparse candidate-table path "stays strict SM100 on purpose ... kernels assert arch == 10" and SM12x gets `dense_blocks` (`:139-145`). For DSV4, `get_platform().is_sm120` also turns `SGLANG_OPT_FP8_WO_A_GEMM` and `SGLANG_OPT_USE_TOPK_V2` off by default (`srt/arg_groups/model_hook.py:491-498`) and forces DeepGEMM metadata for the FP4 indexer because "The SM120 FP4 kernel schedules split_kv=128" (`deepseek_v4_backend.py:1676-1680`).

---

## 8. Issues

**#40286 (OPEN, 2026-09-19) — GLM-5.3-Flash has no usable DSA backend on SM121.** [ISSUE] Reproduction: SGLang v0.5.20 `cu130` arm64, 2x DGX Spark (GB10), TP=2, `nvidia/GLM-5.3-Flash-NVFP4` with DFLASH. Every DSA backend fails: auto (`trtllm`) raises `TllmGenFmhaRunner ... Unsupported architecture` from `trtllm_batch_decode_with_kv_cache_mla`; `tilelang` + FP8 is rejected on CUDA; `tilelang` + BF16 fails with `Failed to set the allowed dynamic shared memory size to 169984`; `triton` is ROCm-only; `flashinfer_sparse_mla` is rejected because the model arch `Glm5NextForConditionalGeneration` is not in `_GLM_DSA_MODEL_ARCHS`. Root cause as stated: no backend covers this arch on SM12x. Proposed fix: admit the arch to the SM12x sparse-MLA path (possibly needing a second page layout) or ship a TileLang config under the smem limit. Response: one contributor comment pointing to issue #39302 as a method that "work[s] in SM120"; no maintainer resolution. [SRC] at the dossier commit `_GLM_DSA_MODEL_ARCHS` still lists only the two `GlmMoeDsa*` names (`flash_mla_sm120.py:29-32`), so the gate described is unchanged.

**#42364 (OPEN, 2026-10-03) — DeepSeek-V4.1 fails at attention init on GB10.** [ISSUE] Reproduction: DeepSeek-V4.1-Flash, TP4 over four DGX Sparks, v0.5.21 and main `00bcc25f6c`; dies in `make_candidate_indexer` with `RuntimeError: the candidate indexer needs DeepGEMM's paged sparse MQA logits`. Root cause as stated: `is_sm100_or_newer()` is `major >= 10` so SM12x skipped the Hopper dense-blocks branch, then required `DEEPGEMM_PAGED_SPARSE_MQA_LOGITS`, which depends on `DEEPGEMM_BLACKWELL` = SM100 only; probe printed `NVIDIA GB10 (12, 1) True False False`. Proposed fix: route SM12x to the Hopper dense-blocks path (64-slot index-K pages). A commenter reproduced on RTX PRO 6000 (SM120) and reported a backport of PR #42371's `__init__.py` hunk: CUDA-graph replay test 0/2 -> 2/2. No maintainer comment. [SRC] the dossier commit contains the fix shape: `if not is_sm100() or candidate_topk_blocks <= 0: return dense_blocks, dense_blocks` with a comment citing #49173 (`v41_indexer/__init__.py:139-145`). **[INF]** the startup failure is fixed on main although the issue is still open.

**#50749 (OPEN, 2026-11-16) — SM120 BF16 TileLang DSA decode smem overflow.** [ISSUE] Reproduction: RTX 5090, standalone call to `tilelang_kernel.tilelang_sparse_fwd` with `d_v=512, d_tail=64, heads=8, topk=2048`; `tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 170048`, while the GPU reports `shared_memory_per_block_optin=101376`. In serving, `max_tokens=1` works and `max_tokens=2` crashes at the first decode step. Root cause as stated: `sparse_attention_fwd_kernel_v2` with default `block_I=64` needs `2 * 64 * (512 + 64) * 2 = 147456` bytes for the two KV buffers alone. Setting `block_I=32` exposes a second bug: loaders use `T.serial(4)` with row index `r * 16 + (tx - 256) // 8`, addressing rows 0..63 of a 32-row buffer -> `CUDA_ERROR_ILLEGAL_ADDRESS`. Proposed fix: a configuration that fits the per-block limit. No comments. One day old at the dossier commit; **[INF]** unfixed.

**#33422 (OPEN, 2026-08) — DSV4 prefill slower on 4x RTX PRO 6000 than vLLM/Marlin.** [ISSUE] Reporter: DeepSeek-V4-Flash-0731, TP=4, v0.5.16, `dsv4` + `flashinfer_mxfp4` + FP8 KV: prefill ~5,540 tok/s (9x90K prompts), ~2,805 tok/s (3x650K), vs a vLLM 0.25.1 recipe "~12.5K tok/s at 256K context". Suspected causes: unoptimized SM120 dsv4 prefill, MoE kernel choice, disabled prefill CUDA graph. Responses: a commenter pointed to PR #29927 (first shipped in v0.5.20: batched sparse-MLA prefill, DeepGEMM indexer, DeepGEMM FP4 MoE on SM120). A contributor re-ran on v0.5.22 on 4x RTX PRO 6000 **Server Edition**: vLLM recipe 8,171 tok/s vs SGLang 8,983 (9x90K), 7,150 vs 7,728 (1x238K); `--ep-size 4` +3%; DP attention + EP4 reached 11,756 on 9x90K but 6,188 on a single long prompt; Marlin about 5% slower than `flashinfer_mxfp4`; the 12.5K vLLM figure did not reproduce. Caveat stated by that contributor: engines ran on different hosts. No maintainer verdict; issue still open.

---

## 9. Docs page `docs/docs/advanced_features/attention_backend.mdx`

All text mentioning SM120 / 5090 / Blackwell / Spark [SRC]:

- `:240` MLA table row label "**TRTLLM MLA (Blackwell)**"; `:249` "**CuteDSL MLA (Blackwell)**".
- `:390` "**Blackwell SM120 (e.g., RTX PRO 6000 Blackwell)**: `triton` or `flashinfer` for prefill/full attention; `trtllm_mha` is supported for `--decode-attention-backend` only."
- `:391` "**Other Blackwell variants (including SM100 B200/GB200)**: `triton`, `trtllm_mha`, or `fa4` only."
- `:423` DSA `flashmla_sparse`: "Default prefill on Hopper and Blackwell (BF16)".
- `:435` DSA `triton_sparse_mla`: "... CUDA SM90+ (tiles tuned on SM90 and SM120; other architectures run an untuned tile and log a warning at startup), at most 32 query heads per rank, not with `index_kpool`."
- `:459` DSA `trtllm`: "Default decode on Blackwell (BF16); default for FP8 on Blackwell (prefill + decode)".
- `:512` "# Example: Prefill with FA4, Decode with TRTLLM MLA (Blackwell)".
- `:549` "**Blackwell (e.g., B200)**: Defaults to `trtllm_mha`, unless using speculative decoding with `topk > 1`."
- `:554` "**Blackwell**: Defaults to `flashinfer`; `trtllm_mla` is auto-selected for DeepSeek V3 models specifically."
- `:613` "TRTLLM MLA (Optimized for Blackwell Architecture, e.g., B200)"; `:632` "TRTLLM MHA (Optimized for Blackwell Architecture, e.g., B200)".
- `:641-642` "TRTLLM MHA (XQA backend) (Optimized for SM90 and SM120, e.g., H20, H200, 5090) / Note that TRTLLM XQA backend only works well for pagesize 64."
- `:674` "# FA4 for both prefill and decode on SM90/SM100".
- **"Spark", "GB10", "SM121", "SM12x": not found** in this page.

**MHA matrix** (`:19-187`):

| Backend | Page Size > 1 (native) | FP8 KV | FP4 KV | Spec topk=1 | Spec topk>1 | Sliding Window | MultiModal |
|---|---|---|---|---|---|---|---|
| FlashInfer | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ |
| FA3 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| FA4 | 128 | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Triton | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Torch Native (SDPA) | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| FlexAttention | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| TRTLLM MHA | 16, 32 or 64 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Dual Chunk FlashAttention | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| HPC-Ops | 64 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| AITER (ROCm) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Wave (ROCm) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Ascend (NPU) | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ |
| Intel XPU | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Intel AMX (CPU) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**MLA matrix** (`:188-300`):

| Backend | Native Page Sizes | FP8 KV | FP4 KV | Chunked Prefix Cache | Spec topk=1 | Spec topk>1 |
|---|---|---|---|---|---|---|
| FlashInfer MLA | 1 | ❌ | ✅ | ✅ | ✅ | ❌ |
| FlashMLA | 64 | ✅ | ✅ | ✅ | ✅ | ❌ |
| Cutlass MLA | 128 | ✅ | ✅ | ✅ | ✅ | ❌ |
| TRTLLM MLA (Blackwell) | 32 or 64 | ✅ | ✅ | ✅ | ✅ | ❌ |
| CuteDSL MLA (Blackwell) | 32 or 64 | ✅ | ❌ | ✅ | ✅ | ❌ |
| FA3 | n/a | ❌ | ❌ | ✅ | ✅ | ⚠️ (page_size=1 only) |
| Triton | n/a | ❌ | ❌ | ❌ | ✅ | ⚠️ (page_size=1 only) |
| FA4 | 1 | ❌ | ✅ | ✅ | ❌ | ❌ |
| Ascend MLA (NPU) | 128 | ❌ | ❌ | ❌ | ❌ | ❌ |

**Docs vs code mismatches on SM12x:** (a) FA4 "128" page size is SM100-only in code (section 1.6); (b) TRTLLM MHA "16, 32 or 64" vs code accepting 128 (section 3); (c) `:549` "Blackwell defaults to `trtllm_mha`" is true only for SM100, SM12x defaults to `flashinfer`; (d) `:554` MLA "Blackwell defaults to `flashinfer`" is SM100-only, SM12x defaults to `triton`; (e) hybrid GDN `trtllm_mha` decode-only is not enforced by the assertion.

---

## 10. Operator cheat-sheet for SM12x (flags from source; none executed here)

```bash
# A. Default dense MHA path (auto): flashinfer fa2; triton if the model has attention sinks
python3 -m sglang.launch_server --model-path <MHA model>

# B. FlashInfer explicitly
... --attention-backend flashinfer

# C. Triton explicitly (required for sinks unless fa4/trtllm_mha is chosen)
... --attention-backend triton

# D. SGLang SM12x FA4, both phases. BF16/FP16 KV only; head_dim <= 256; not for MLA or hybrid GDN.
#    Page size is NOT auto-forced on SM12x; 64 enables the page-TMA HD256 decode specialization.
... --attention-backend fa4 --page-size 64
#    Prefill = 1 split, decode = auto split. Deterministic (all splits = 1):
... --attention-backend fa4 --enable-deterministic-inference

# E. FA4 prefill + XQA decode (page size snaps to 64; BF16 KV)
... --prefill-attention-backend fa4 --decode-attention-backend trtllm_mha --page-size 64

# F. TRTLLM MHA both phases: FMHAv2 prefill + XQA decode.
#    Do NOT combine with --kv-cache-dtype fp8_e4m3 or SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR>0.
... --attention-backend trtllm_mha --page-size 64

# G. FP8 KV with XQA decode only (prefill on flashinfer or triton)
... --prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha \
    --kv-cache-dtype fp8_e4m3 --page-size 64

# G2. NVFP4 KV: pass the dtype only; the resolver sets prefill=flashinfer (FP8 dequant) + decode=trtllm_mha (XQA)
... --kv-cache-dtype nvfp4

# H. Speculative decoding: trtllm_mha needs topk 1; topk>1 with page_size>1 needs flashinfer/triton
... --speculative-eagle-topk 1 --speculative-attention-mode decode

# I. Hybrid GDN models (Qwen3.5 / Qwen3-Next class): allowed set on SM12x = triton | trtllm_mha | flashinfer
... --attention-backend flashinfer --decode-attention-backend trtllm_mha --page-size 64

# J. DeepSeek-V4 (attention_backend=dsv4 and page_size=256 are set by the model override)
SGLANG_SM120_FLASHMLA_BACKEND=flashinfer   # default; "triton" = fused Triton; anything else = PyTorch
SGLANG_OPT_SM120_DIRECT_SWA_KV=1           # optional: store SWA KV in 64-token pages
... --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_mxfp4

# K. DSA models: GLM (GlmMoeDsaForCausalLM) with FP8 KV auto-selects flashinfer_sparse_mla
... --kv-cache-dtype fp8_e4m3            # or explicitly:
... --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla
#    Optional validated alternative for prefill: --dsa-prefill-backend triton_sparse_mla (<= 32 q heads/rank)

# L. Generic MLA (DeepSeek V3 class): default triton
... --attention-backend triton
```

Debug aid: `FLASH_ATTENTION_ARCH=sm_120` overrides the FA4 arch detection (`interface.py:86-101`).

---

## 11. Assumptions, open points, and discriminating experiments

**Not determined:**
1. Line-level differences between `fa4_sm120/flash_fwd.py` and upstream Dao-AILab `flash_fwd_sm120.py`, and the upstream base commit (upstream source unavailable).
2. What FlashInfer 0.7.4 does internally on SM12x for `backend="auto"` ragged prefill, and how `trtllm_batch_decode_with_kv_cache` selects XQA (FlashInfer source not in tree).
3. Whether dense `trtllm_mla` runs on SM12x (argument layer admits it; issue #40286 suggests the trtllm-gen runner rejects the arch).
4. Whether the SM120-calibrated split-KV model is accurate on GB10 (constants were fitted on 110-SM and 188-SM discrete SKUs).
5. Whether `fa4` prefill + `trtllm_mha` decode is numerically correct end to end.
6. Contents of PR #49044 (reported bug fix in the SM12x kernel).

**Smallest experiments that would falsify the main conclusions:**
- *Page-size claim (1.6):* launch on an SM12x GPU with `--attention-backend fa4` and no `--page-size`; the log must **not** contain "FA4 backend only supports page size 128". If the warning appears, the `is_sm100` gate reading is wrong.
- *FP8 claim (1.8):* `--attention-backend fa4 --kv-cache-dtype fp8_e4m3` on SM12x must fail with `FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe.`
- *SM121 routing (5):* on a DGX Spark, `python3 -c "from sglang.srt.utils.device import is_sm120, is_sm120_supported, is_sm121; print(is_sm120(), is_sm120_supported(), is_sm121())"` must print `False True True`.
- *smem budget (1.5):* `python3 -c "import cutlass.utils as u; print(u.get_smem_capacity_in_bytes('sm_120'))"` must print `101376`; if it prints a smaller number, HD256 tile (64,64) would not fit and the candidate order would change.
- *Kernel selection (1.6):* run `test/registered/kernels/ops/attention/test_flash_attention_4_sm120.py` on SM121 and compare page size 1 vs 64 HD256 decode latency; equal latency would contradict the page-TMA specialization claim.

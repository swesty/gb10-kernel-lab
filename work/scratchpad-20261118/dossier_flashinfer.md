# FlashInfer attention kernels that can run on SM120 / SM121 — source dossier

Source tree: FlashInfer `main` @ `778f57e4d73a73a439ad9cb88daabb5e14cbf816` (commit date 2026-11-17). Read-only, grep/read only; no repository code was executed. All paths are relative to the repository root. Method labels used throughout:

- **[SRC]** — a fact read directly from the cited line(s).
- **[INF]** — an inference made by the author of this dossier from cited source; not stated in the source.
- **[ND]** — not determined / not found.

Abbreviations: "b12x paged" = `flashinfer/experimental/b12x/attention/paged/`. Inside that directory, file names are given bare (for example `traits.py:101` means `flashinfer/experimental/b12x/attention/paged/traits.py:101`).

---

## 0. Headline conclusions

1. **b12x paged decode stages K/V with TMA (`cp.async.bulk.tensor` through the CuTe `CopyBulkTensorTileG2SOp` atom), using the physical page id as a tensor coordinate; b12x paged extend/prefill stages K/V mostly with per-thread 16-byte `cp.async`.** [SRC] `forward_paged.py:4107-4147`, `forward_paged.py:3782-3792`, `forward_extend_generic.py:3272-3290`, `forward_extend_generic.py:3836-3870`.
2. **Every FlashInfer attention path admitted on SM12x computes with warp-level `mma.sync`; none uses `wgmma` or `tcgen05`.** [SRC] per-kernel citations in sections 1–5; `flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/README.md:17-19` states the reason explicitly.
3. **Split-KV exists for decode in b12x paged, FA2, and XQA, and all three size the split from the device SM count.** [SRC] `planner.py:216-228`, `include/flashinfer/attention/scheduler.cuh:175-205`, `csrc/xqa/mha.cu:3123-3129`. b12x paged extend never splits KV. [SRC] `planner.py:2038-2039`, `planner.py:2277-2280`.
4. **b12x paged decode rejects `head_dim == 64` because of two independent trait rules (section 1.9).** The GitHub issue number 6977 does not appear anywhere in the tree ([ND]); the mechanism below is [INF] from the trait code.
5. **SM121 is admitted by b12x (`cap[1] in (0, 1)`), by FA2, by XQA (`__CUDA_ARCH__ == 1210`), and by FMHAv2; several b12x "Laguna" tile specializations are keyed on exactly `(12, 0)` and therefore do not apply on SM121.** [SRC] `flashinfer/experimental/b12x/_lib/gating.py:28-31`, `traits.py:383`, `traits.py:411`, `traits.py:459`, `traits.py:468`, `traits.py:486`.

---

## 1. b12x paged attention (`flashinfer/experimental/b12x/attention/paged/`)

### 1.1 File list

[SRC] (line counts from `wc -l`): `__init__.py` (115, `META`), `api.py` (101, public surface), `_controls.py` (62, env-var controls), `_forward.py` (1980, dispatch/compile/launch), `_preparation.py` (569), `_scratch.py` (2812), `_selected_forward.py` (1608) + `_selected_forward_config.py` (8), `_tuning.py` (465), `forward_paged.py` (11267, decode/verify kernels), `forward_extend_generic.py` (7621, extend/prefill kernel), `graph_replay.py` (2381, on-device schedule rebuild for CUDA graphs), `merge.py` (1368, split-KV LSE merge), `planner.py` (2601), `reference.py` (419), `traits.py` (580), `workspace.py` (2353).

Provenance: the op declares upstream `https://github.com/lukealonso/b12x` at commit `6627d342`, `since="0.7.0"`, `requires=("triton",)`. [SRC] `__init__.py:69-79`.

### 1.2 Public API and entry points

- **`get_b12x_paged_attention`: not found.** A repository-wide grep for `get_b12x_paged_attention`, `get_b12x`, and `b12x_paged` over `*.py`, `*.md`, `*.cuh`, `*.rst` returns no match. [ND] The symbol presumably lives on the vLLM side. Within FlashInfer core, `b12x` is referenced only by the MoE and GEMM paths (for example `flashinfer/__init__.py:201` exports `b12x_fused_moe`); no core attention dispatch references b12x paged attention.
- The public module is `b12x.attention.paged` with a "planned" lifecycle: `plan(caps, invocation=...)` → `PreparationSession.prepare` → `bind(plan, ...)` → `run(binding=..., plan=...)`. [SRC] `__init__.py:10-28`, `api.py:38-68`.
- Exported names: `Caps`, `Plan`, `Binding`, `Workspace`, `Budget`, `DecodeGraphCapacity`, `ExtendGraphCapacity`, `VerifyGraphCapacity`, `DecodeGraphScratchEnvelope`, `decode_graph_capacity`, `extend_graph_capacity`, `verify_graph_capacity`, `plan`, `bind`, `compile`, `run`, `infer_mode`, `is_supported`, `clear_caches`. [SRC] `api.py:75-100`.
- `extend_graph_capacity` is `planner.plan_extend_graph_capacity` (`api.py:32`). It returns `PagedExtendGraphCapacity(graph_ctas_per_sm, cta_tile_q, max_work_items, max_effective_kv_pages, representative_cache_seqlen)`. [SRC] `planner.py:979-992`, `planner.py:1587-1593`. It requires `page_size in (64, 128)` (`planner.py:1523-1524`) and computes `max_work_items = max(num_sms * graph_ctas_per_sm // num_kv_heads, total_num_qo_tiles)` (`planner.py:1579-1586`, `planner.py:227-228`).
- `is_supported(device)` delegates to `default_is_supported`, which requires compute capability major 12 and minor 0 or 1, `nvidia-cutlass-dsl >= 4.6.0`, and Triton. [SRC] `api.py:71-72`, `_lib/gating.py:14`, `_lib/gating.py:28-31`, `_lib/gating.py:89-95`:

```python
def is_b12x(device=None) -> bool:
    """True when the target device is consumer Blackwell (SM120/SM121)."""
    cap = get_compute_capability(device)
    return cap is not None and cap[0] == 12 and cap[1] in (0, 1)
```

- **Work-item concept.** A plan is a flat list of work items; each item is (request index, Q-tile index, KV-chunk index) plus merge/output index pointers. [SRC] `planner.py:1834-1842` (`request_indices`, `qo_tile_indices`, `kv_tile_indices`, `merge_indptr`, `o_indptr`, `block_valid_mask`, `kv_window_start_tokens`). The decode grid is `(num_work_items, num_kv_heads, 1)`: one CTA per (work item, KV head). [SRC] `forward_paged.py:4178`.
- Modes: `decode` (every request has q_len 1), `extend` (anything else), and an explicit `verify` mode for multi-token speculative verification. [SRC] `planner.py:788-792`, `planner.py:72` (`_PagedMode = Literal["decode", "extend", "verify"]`).

### 1.3 MMA atom

- BF16 path: raw inline `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, issued twice per call to form an "m16n16k16" step (helper `bf16_mma_m16n16k16_f32`). [SRC] `flashinfer/experimental/b12x/_lib/intrinsics.py:3920-3961`, `intrinsics.py:3985`.
- FP16 path: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. [SRC] `intrinsics.py:3699`.
- Operands are loaded with `ldmatrix` (`ldmatrix_m8n8x4_b16`, left/right-half variants, and a byte variant `ldmatrix_m16n16x1_trans_b8`). [SRC] `forward_paged.py:49-55`, `forward_paged.py:1657-1682`.
- Optional native FP8: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0` (`mxfp8_mma_m16n8k32_f32_e4m3`). [SRC] `intrinsics.py:4019`, `intrinsics.py:4083`. A plain `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` helper also exists at `intrinsics.py:4311`.

### 1.4 Warps, threads, tiles

Trait functions [SRC] `traits.py:35-44`:

```python
def paged_get_num_warps_q(cta_tile_q: int) -> int:
    return 4 if cta_tile_q > 16 else 1
def paged_get_num_warps_kv(cta_tile_q: int) -> int:
    return 4 // paged_get_num_warps_q(cta_tile_q)
def paged_get_num_mma_q(cta_tile_q: int) -> int:
    return 2 if cta_tile_q > 64 else 1
```

- `num_threads = num_warps_q * num_warps_kv * 32` = 128 in the general path (`traits.py:348`); `can_implement` rejects anything other than 128 threads (`forward_paged.py:3962-3963`). Launch block is `[32, num_warps_q, launch_warps_kv]` (`forward_paged.py:4207`). "Role-specialized" decode adds one extra warp (160 threads) that acts as a dedicated TMA producer (`forward_paged.py:3268-3270`). One hard-coded exception: FP8 KV with `cta_tile_q == 48` uses 3 Q warps, 96 threads, `cta_tile_kv = 32`, payload 49152 B (`traits.py:142-196`).
- `cta_tile_kv = num_mma_kv * num_warps_kv * 16` (`traits.py:289`).
- **Decode / verify**: `cta_tile_q` is forced to 16 (`planner.py:826-832`), so the layout is 1 Q warp × 4 KV warps. The generic candidate sweep allows only `num_mma_kv = 1` when `cta_tile_q == 16` (`traits.py:272-276`), giving a **16 Q rows × 64 KV rows** tile. Exact specializations can request `num_mma_kv = 2` (16 × 128) (`traits.py:464`, `traits.py:483`).
- **Extend**: `cta_tile_q` is 16 when packed Q length ≤ 32 (`planner.py:833-835`), otherwise the FA2 rule: 128 if packed length > 64 and `head_dim < 256`, else 64 if > 16, else 16 (`planner.py:795-801`). For 64/128 the layout is 4 Q warps × 1 KV warp and `cta_tile_kv = 16 * num_mma_kv` with `num_mma_kv ≤ 8 / num_mma_q` (`traits.py:225`).
- Head dims: `head_dim_qk` and `head_dim_vo` must be multiples of 16 at trait level (`traits.py:121-122`), multiples of 32 in `can_implement` (`forward_paged.py:3960-3961`); the decode TMA path further requires BF16 KV dims to be multiples of 64 and ≤ 256, and FP8 KV to have `head_dim_qk` in [128, 256] step 64 and `head_dim_vo` a multiple of 128 (`forward_paged.py:3325-3339`).

### 1.5 Shared memory budget and stage count

- The device limits are read at runtime, not hard-coded: `max_smem_per_sm = shared_memory_per_multiprocessor`, `max_smem_per_block = min(max_smem_per_sm, shared_memory_per_block_optin)`. [SRC] `traits.py:204-211`. No literal 99 KB constant appears in b12x paged. (The b12x MLA family does hard-code `SM120_SMEM_CARVEOUT_BYTES = 101376`, "(100-1)*1024", at `flashinfer/experimental/b12x/attention/_shared/mla/smem.py:64-66`.)
- Payload = `align16(max(Q + num_mma_kv * kv_bytes_per_mma, sync storage + compact stage, O stage, minimum))` where `kv_bytes_per_mma = (upcast_stride_k + upcast_stride_v) * 16 * 16 * num_warps_kv`. [SRC] `traits.py:223-224`, `traits.py:290-299`.
- The launch size is the `size_in_bytes()` of a CUTLASS struct holding two mbarrier arrays (`Int64[2 * num_stages]` each) plus the payload aligned to 1024. [SRC] `flashinfer/experimental/b12x/_lib/smem.py:34-43`.
- Residency: a candidate gets 2 CTAs/SM if `2 * launch_bytes <= max_smem_per_sm`, else 1; the selection maximizes `(residency, num_mma_kv)` — occupancy first, tile width second. [SRC] `traits.py:308`, `traits.py:329`. The launch passes `min_blocks_per_mp=self.num_ctas_per_sm` (`forward_paged.py:4210`).
- **[INF] worked numbers** (my arithmetic from `traits.py:213-256`): BF16 decode, `head_dim` 128, `cta_tile_q` 16 → Q = 4096 B, K+V stage = 32768 B (64 rows × 256 elems × 2 B), sync storage = 4·16·(128+24)·4 + 4·16·8 = 39424 B → payload 39424 B. Same with `head_dim` 256 → Q 8192 + K/V 65536 = 73728 B payload. FP8 KV, `head_dim` 128 → K/V stage 16384 B. All fit under a 99 KB per-block limit; the 256-dim BF16 case cannot be 2-resident on a ~100 KB SM.
- **Stages**: 1 stage whenever `num_warps_kv > 1` or KV is FP8; 2 stages only for single-KV-warp BF16 when `Q + 2·KV` fits; 3 stages only for an opt-in BF16 "minimax" decode specialization. [SRC] `forward_paged.py:3271-3285`. The general decode class raises `NotImplementedError` if `num_stages != 1` and the kernel is not role-specialized (`forward_paged.py:3383-3386`). Extend: same 1-or-2 rule (`forward_extend_generic.py:3263-3271`).

### 1.6 K/V staging from the paged cache

**Decode / verify (`PagedForwardKernel` in `forward_paged.py`) — TMA only.** The constructor raises unless the exact-plane TMA path is selected: "PagedForwardKernel now only supports exact-plane paged K/V TMA decode; extend and legacy non-TMA ingress use dedicated specialized kernels." [SRC] `forward_paged.py:3376-3382`.

Mechanism [SRC]:
1. The cache `[num_pages, page_size, kv_heads, head_dim]` is re-viewed with modes permuted to `[page_size, head_dim, kv_heads, num_pages]` (`forward_paged.py:4077-4083`), so the page id becomes the last tensor coordinate.
2. A tiled TMA atom is built with `cpasync.CopyBulkTensorTileG2SOp()` and `make_tiled_tma_atom` over a tile of `(stage_tile_rows, kv_tma_plane_head_dim)` (`forward_paged.py:4107-4147`). The plane width is 64 elements for BF16 and 128 bytes for FP8 — 128 bytes per row either way (`forward_paged.py:3427-3430`); a `head_dim` is covered by 1–4 planes (`forward_paged.py:3432-3444`). The smem layout uses `make_swizzle(3, 4, 3)` (`forward_paged.py:3514`).
3. The block-table lookup happens per stage tile, in one thread, immediately before the TMA issue:

```python
page_id = (
    Int32(0)
    if const_expr(paged_control("B12X_PAGED_KV_TMA_FORCE_PAGE0", "0") == "1")
    else mPageTable[request_idx, page_idx]
)
pipeline_tma.producer_acquire(producer_state)
src_idx = self._tile_src_idx_from_page_id(page_id, sub_tile)
load_tma0(src_idx=src_idx, producer_state=producer_state)
```
   `forward_paged.py:3782-3789`; `page_idx = tile_token_base // page_size` (`forward_paged.py:3670`).
4. Page boundaries never cut a TMA tile: the constructor requires `page_size % cta_tile_kv == 0` (`forward_paged.py:3159-3163`). For `page_size == 128` a page is addressed as `page_id * page_tiles_per_entry + sub_tile`, widened to `Int64` (`forward_paged.py:3763-3766`); `page_tiles_per_entry` is derived from the cache *stride* so vLLM's combined `[N, 2, 128, H, D]` cache works (`forward_paged.py:117-126`).
5. Pipelines are `PipelineTmaAsync` for K and V with mbarrier storage in smem and `tx_count` equal to the staged byte count (`forward_paged.py:4737-4752`). TMA descriptors are prefetched (`forward_paged.py:4559-4560`).
6. **Producer**: in the default kernel there is no dedicated producer warp — warp 0 (`warp_linear_idx == 0`) issues the next tile's TMA after releasing the consumer stage (`forward_paged.py:5598`, `forward_paged.py:6350-6366`). In "role-specialized" decode an extra warp runs a producer loop over the whole chunk, calls `setmaxregister_decrease(80)`, then exits; math warps call `setmaxregister_increase(168|248)` (`forward_paged.py:5236-5335`).
7. Q is staged with predicated 16-byte `cp.async.cg.shared.global.L2::128B` (`forward_paged.py:890-914`, `forward_paged.py:5343-5361`).

**Supported page sizes: 64 and 128 only.** [SRC] `planner.py:1945-1948`, `forward_paged.py:3151-3154`.

**Extend (`forward_extend_generic.py`)**: TMA is enabled only for one narrow shape (BF16, `head_dim` 256/256, `cta_tile_q` 16, 64-row stage, `page_size` 64, 1 stage) — `forward_extend_generic.py:3275-3290`, with the comment "Page-128 uses the byte-addressed copy path until extend TMA gets the decode kernel's page-entry flattening" (`forward_extend_generic.py:3272-3274`). Otherwise every thread computes a per-row page-table lookup and issues 16-byte `cp.async` loads (`forward_extend_generic.py:3836-3870`). No dedicated producer warp in extend. [SRC by absence of a role branch; treat as INF.]

### 1.7 Split-KV for decode, reduction, and the split heuristic

- Decode auto-splits only when `enable_cuda_graph and graph_chunk_policy`, Q is BF16, and the plan is not MSA (`planner.py:2024-2033`); verify always splits (`planner.py:2035`); extend never splits (`planner.py:2038-2039`).
- The merge backend supports split-KV only for output dtype fp16/bf16/fp32 **and `head_dim_vo in (128, 256)`** (`planner.py:75-84`); otherwise split-KV is disabled (`planner.py:2265-2271`).
- Partials are written to `workspace.tmp_output` / `workspace.tmp_lse` (`_forward.py:1195-1196`) and reduced by `PagedPersistentMergeKernel`: partial outputs are normalized, partial scores are **base-2 LSE**, CTAs walk `(row, head)` work persistently (`merge.py:1-14`, `merge.py:198-223`).
- **The heuristic reads the SM count.** Work-item capacity is `num_sms * graph_ctas_per_sm // num_kv_heads` (`planner.py:227-228`); chunks per request = that capacity divided by `batch * query_tiles_per_request` (`planner.py:244-245`). `graph_ctas_per_sm` comes from a shape-keyed table (1, 2, 4, or 6; default 2) (`planner.py:493-548`), and nominal chunk budgets are table-driven by (head_dim, GQA group, KV dtype, batch) — 192, 96, 48, 64, or 128 total chunks (`planner.py:697-749`). Chunk pages = `ceil(max_effective_kv_pages / max_chunks)` (`planner.py:785`). Merge CTA count is `num_sms * min(3, ceil(total_work / num_sms))` (`merge.py:170-175`).
- Non-graph path: a binary search over KV chunk size identical in form to FA2's (`planner.py:1037-1086`), with `min_kv_chunk_size = max(128 // page_size, 1)` (`planner.py:2276`).

### 1.8 FP8 KV, sinks, sliding window, GQA

- **FP8 KV default = dequantize after load, BF16 MMA.** FP8 bytes are TMA-staged raw, `ldmatrix`-loaded, converted in registers (`fp8x4_e4m3_to_bfloat2x2`), then fed to the BF16 `m16n16k16` helper. [SRC] `forward_paged.py:1680-1709`. An SM120-only helper uses `cvt.rn.bf16x2.e4m3x2` ("requires PTX ISA 9.2 and an architecture-specific SM120 target") — `_lib/intrinsics.py:3163-3180`.
- **Native FP8 MMA is opt-in** via env `B12X_TURBO_ATTN=1` (`_forward.py:78-79`, `_forward.py:164-168`), disabled for MSA, and for decode further limited by batch/chunk thresholds (`_forward.py:175-192`). FP8 KV requires BF16 queries (`traits.py:127-128`).
- Scales: `k_descale` / `v_descale` of shape `(batch,)` or `(batch, kv_heads)`, read per request (`forward_paged.py:4020-4021`, `forward_paged.py:5403-5420`).
- **Sinks**: `attention_sink_bias[q_head]` is folded into the running (m, d) state as an extra logit in base-2, by KV warp 0, and with split-KV only by the chunk that owns the causal limit (`forward_paged.py:977-1011`).
- **Sliding window**: `window_left` (−1 = full). The planner converts it to a per-request window start page (`planner.py:198-213`, `planner.py:2070-2086`); the kernel reads `mKvWindowStartTokens[request_idx]` (`forward_paged.py:4458-4462`). MSA block-sparse plans reject `window_left` (`planner.py:1921-1924`).
- **GQA packing**: Q rows of the heads that share one KV head are packed as `packed_q_idx = token * group_size + head_in_group`; `q_head_idx = kv_head_idx * group_size + q_group_lane` (`forward_paged.py:3909-3920`). One CTA serves one KV head and all of its query heads; a decode request with group size ≤ 16 fills part of one 16-row M tile.

### 1.9 Trait rules and why `head_dim == 64` fails

[SRC] `traits.py:88-105`:

```python
kv_is_fp8 = kv_dtype == _FP8_KV_DTYPE
if num_mma_d_vo < 4:
    return True
if num_mma_d_vo == 4 and num_mma_kv % 2 == 1:
    return True
if num_mma_q * (8 * num_mma_d_vo + 8 * num_mma_kv) >= 256:
    return True
if kv_is_fp8 and (num_mma_kv * 2) % num_warps_q != 0:
    return True
```

[INF] Causal chain for decode with `head_dim` 64: `num_mma_d_vo = 64 / 16 = 4` (`traits.py:199`) → decode forces `cta_tile_q = 16` (`planner.py:826-832`) → the candidate sweep for `cta_tile_q == 16` only tries `num_mma_kv = 1` (`traits.py:272-276`) → rule 2 (`num_mma_d_vo == 4 and num_mma_kv % 2 == 1`) rejects that sole candidate → `ValueError("no valid NUM_MMA_KV typed shared-memory allocation fits the device")` (`traits.py:319-322`). Independently, split-KV merge requires `head_dim_vo in (128, 256)` (`planner.py:75-84`), and FP8 KV on the decode TMA path requires `head_dim_qk >= 128` (`forward_paged.py:3332-3339`). The same rule 2 exists in core FA2 (`include/flashinfer/attention/prefill.cuh:467`), but FA2's dispatcher may pick an even `NUM_MMA_KV`, so FA2 is not blocked the same way. Unresolved: whether issue #6977's reported error text matches `traits.py:320-322`; the issue is not in the tree.

### 1.10 SM121 vs SM120 gating; fences and mbarriers

- No `sm_121` / `(12, 1)` literal occurs in the paged directory (grep). [SRC by absence] Five trait specializations and one planner budget are keyed on `device_capability == (12, 0)` (`traits.py:383`, `411`, `459`, `468`, `486`; `planner.py:272`); two use `device_capability[0] >= 12` (`traits.py:440`, `traits.py:533`). [INF] On SM121 the generic trait sweep is used in place of the Laguna-specific tiles.
- `fence_proxy`: no occurrence in `attention/paged/*.py` or `attention/_shared/cute/*.py` (grep). [SRC by absence] It is used elsewhere in b12x (MoE, GEMM, quantization kernels).
- mbarriers: K and V barrier arrays live in smem (`_lib/smem.py:34-38`); decode uses `PipelineTmaAsync` (`forward_paged.py:4737-4752`) and a manual `mbarrier_arrive_and_expect_tx` under `elect_one` in the manual issue helpers (`forward_paged.py:335-347`); extend calls `mbarrier_init` / `mbarrier_wait` directly (`forward_extend_generic.py:4443-4445`, `5559`).

---

## 2. FA2-style template attention on SM12x

- **Dispatch.** `determine_attention_backend` returns `"fa3"` only when `is_sm90a_supported(device)` (major == 9); everything else, including SM120/SM121, gets `"fa2"`. [SRC] `flashinfer/utils.py:717-730`, `flashinfer/utils.py:766-768`.
- Arch helpers [SRC] `flashinfer/utils.py:789-822`: `is_sm120a_supported` = major 12, minor 0, CUDA ≥ 12.8; `is_sm121a_supported` = major 12, minor 1, CUDA ≥ 12.9; `is_sm12x_supported` = major 12 with min CUDA 12.9 if minor ≥ 1 else 12.8; `is_sm120f_supported` = major 12, CUDA ≥ 12.9.
- **Automatic upgrade to FMHAv2**: in ragged prefill planning, an `"fa2"` selection is rewritten to `"fmha_v2"` on SM120/SM121 when the problem is MHA (`num_qo_heads == num_kv_heads`), NHD, symmetric head dim in {64, 128, 256}, same Q/KV dtype (fp16/bf16), `use_fp16_qk_reduction`, `max_seq_in_batch >= 256`, no window, no softcap. [SRC] `flashinfer/prefill.py:7016-7044`, `flashinfer/utils.py:653-672`.
- **Tile sizes.** `FA2DetermineCtaTileQ`: 128 if packed Q length > 64 and `head_dim < 256`; else 64 if > 16; else 16 (subject to a smem-fit check); 16/32 for `head_dim >= 512`. [SRC] `include/flashinfer/utils.cuh:411-450`. `CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16`, `NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * 32`. [SRC] `include/flashinfer/attention/prefill.cuh:356`, `prefill.cuh:389`.
- **Smem-driven `NUM_MMA_KV`.** The dispatcher reads `cudaDevAttrMaxSharedMemoryPerMultiprocessor` and `cudaDevAttrMaxSharedMemoryPerBlockOptin`, sets `num_ctas_per_sm = 2` if two minimum tiles fit, clamps the per-block budget, and dispatches the largest `NUM_MMA_KV` that fits both smem and the register bound `8 / NUM_MMA_Q`. [SRC] `prefill.cuh:5005-5010`, `prefill.cuh:5069-5107`. A comment names SM120 as a tight-smem part (`prefill.cuh:5046-5050`).
- **Staging: `cp.async` only.** `page_produce_kv` issues `load_128b_async` per thread with a predicate `kv_idx < kv_len` (`prefill.cuh:691-739`); the non-tensor-core decode kernel uses `cp_async::pred_load` with `commit_group` / `wait_group` (`include/flashinfer/attention/decode.cuh:291-362`). MMA is `mma_sync_m16n16k16_row_col_f16f16f32` (`prefill.cuh:1706-1717`).
- **Decode split-KV.** `max_grid_size = num_blocks_per_sm * num_sm` using `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and `cudaDevAttrMultiProcessorCount`; if `batch_size * gdy >= max_grid_size` no split; otherwise binary search the minimum pages per chunk with floor `max(128 / page_size, 1)`; under CUDA graph split is always on. [SRC] `include/flashinfer/attention/scheduler.cuh:175-207`, `scheduler.cuh:74-99`.
- **Prefill split-KV.** `num_blocks_per_sm = 2` hard-coded; `max_batch_size_if_split = (2 * num_sm - num_colocated_ctas) / num_kv_heads`; `PrefillBinarySearchKVChunkSize` searches the smallest chunk whose work-item count fits. [SRC] `scheduler.cuh:795-799`, `scheduler.cuh:101-130`.

---

## 3. XQA decode (`csrc/xqa/`, TRT-LLM port)

- **Arch gate.** JIT: `supported_major_versions=[9, 10, 12]` (`flashinfer/jit/xqa.py:140-142`). C++: SM120 and SM121 are listed with the 99 KB class [SRC] `csrc/xqa/utils.cuh:49-51`:

```cpp
#if __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210
constexpr uint32_t kMAX_SMEM_SIZE = (99u << 10);
```
  with `static_assert(smemSize < kMAX_SMEM_SIZE)` at `csrc/xqa/mha.cu:422`. Auto backend for `trtllm_batch_decode_with_kv_cache`: `"trtllm-gen"` on major 10, else `"xqa"` (`flashinfer/decode.py:4867-4870`; docstring `decode.py:4518`).
- **Head dims**: multiple of 16 in [16, 512]; above 256 no speculative decoding and `head_group_ratio <= 16` (`flashinfer/jit/xqa.py:88-103`). Page sizes 16/32/64/128 (`xqa.py:82-85`).
- **KV dtypes**: fp16/bf16 (enum 0), int8 (1), FP8 e4m3 (2), NVFP4 packed uint8 (3) (`xqa.py:73-80`); NVFP4 KV asserted SM12x-only (`flashinfer/xqa.py:396-399`).
- **Sliding window, sinks, ragged Q**: `SLIDING_WINDOW` compile flag (`jit/xqa.py:108-111`); `sinks` of shape `[num_kv_heads, head_group_ratio]` (`flashinfer/xqa.py:259-262`); ragged Q through `q_cu_seq_lens`, requiring `q_seq_len > 1` (`flashinfer/xqa.py:325-327`).
- **"One query token"**: the kernel is not limited to one token — `q_seq_len > 1` builds `SPEC_DEC=1` (`jit/xqa.py:118-131`). The restrictions found are: the `trtllm_batch_decode_with_kv_cache` xqa branch rejects `cum_seq_lens_q`/`max_q_len`, `return_lse`, and NVFP4 output (`flashinfer/decode.py:4901-4916`); rows per block ≤ 32 in spec-dec (`csrc/xqa/mha.cu:82-85`); and one path asserts `inputSeqLen == 1` (`mha.cu:428`).
- **MMA on SM120**: `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32` (`csrc/xqa/mma.cuh:35-50`). FP8 cache words are converted with `cvt.rn.f16x2.e4m3x2` under `__CUDA_ARCH__ >= 890` (`csrc/xqa/utils.cuh:747-762`); the Hopper GMMA FP8 kernel is selected only when major == 9 (`flashinfer/xqa.py:387-394`). [INF] On SM120 FP8 KV is therefore dequantize-then-HMMA.
- **Layout and staging**: `ctaShapeInWarps = {4, 1, 2}` (8 warps, 256 threads, warp-specialized into two groups), `warpTile = {64, roundUp(nbValidRows, 16)}`, CTA tile 256 KV tokens (`mha.cu:76-102`). On 99 KB archs `preferedKHeadPartBytes = 64` and the V tile is 32 tokens (16 for 16-bit KV with `head_dim > 256`) (`mha.cu:114-117`). Staging is `cp.async` (`ldgsts::copyAsync`, `csrc/xqa/ldgsts.cuh:29-41`) with a per-token page lookup `pageIndices[absoluteTokenIdx / tokensPerPage]` (`csrc/xqa/mhaUtils.cuh:42-43`).
- **Split-KV ("multi-block mode")**: `nbSubSeqPerSeq = min(max(1, multiProcessorCount / (batchSize * nbKHeads)), divUp(maxSeqLen, ctaTile.x))` (`mha.cu:3123-3129`); the SM count is passed from Python (`flashinfer/decode.py:5585`). Sinks are applied in-kernel only when not multi-block (`mha.cu:2618`).
- A separate MLA kernel `csrc/xqa/mla_sm120.cu` is built only for major 12 (`jit/xqa.py:222`), is 384 threads (`__launch_bounds__(32 * 4 * 3, 1)`), and stages with TMA (`tma::loadAsync` with a 4-D coordinate whose last index is the page id, `mla_sm120.cu:132`).

---

## 4. FMHAv2 prefill (TRT-LLM `fmha_v2`)

- **Arch gate**: `gen_fmha_v2_module` compiles for `supported_major_versions=[9, 12]` (`flashinfer/jit/attention/modules.py:2667-2669`); the dedicated SM120 module uses `[12]` (`modules.py:2606-2608`). `fmha_v2_prefill_sm120` requires capability in `((12, 0), (12, 1))` and its docstring says "validated for SM120 and SM121" (`flashinfer/prefill.py:9915`, `prefill.py:9939-9942`).
- **Head dims**: SM120 generic configs `[64, 128, 256, 512]` (`flashinfer/jit/attention/fmha_v2/fmha_library.py:1294`); 512 uses "q_loop_step = kv_loop_step = 64, sm_mma=80, tiled noloop" (`fmha_library.py:351-355`). `fmha_v2_prefill_sm120` accepts 64/128/256 (`prefill.py:9963-9970`). Kernel list includes `..._e4m3_fp32_64_64_S_q_k_v_{64x64,128x128,192x128,256x256}_output_bf16_sm120` and `bf16_64_128_S_q_k_v_192x128_sm120` (`modules.py:2589-2597`).
- **FP8 limitation**: the ragged entry `trtllm_fmha_v2_prefill` raises "FP8 (e4m3) is not yet supported for FMHAv2 on SM120 (Blackwell)" (`prefill.py:10443-10448`). The fixed-shape `fmha_v2_prefill_sm120` entry *requires* FP8 e4m3 Q/K/V with BF16 output (`prefill.py:9945-9946`, `prefill.py:9975`), contiguous BSHD, same seq_len for every batch item (`prefill.py:9905-9907`).
- **Sliding window**: supported; chunked-causal is rejected on SM12x because "the non-warp-specialized 'tiled' kernels used on SM120 honor `sliding_window_size` only" (`prefill.py:10410-10421`).
- MMA family: comments describe "HMMA, Ampere-compatible" kernels (`flashinfer/jit/attention/fmha_v2/generate_kernels.py:47`, `flashinfer/utils.py:662-664`); FP8 variants are enumerated through `enumerate_qmma_flash_kernels` (`generate_kernels.py:52-69`). Staging mechanism inside the generated kernels: [ND] (not read).

---

## 5. Sparse MLA / cake kernels for SM120

- **Cake DeepSeek-V4 NVFP4 sparse MLA (`csrc/cake_dsv4/sm_120a/`, JIT `flashinfer/jit/cake_sparse_mla_sm120_dsv4_nvfp4.py`, `cake_sparse_mla_sm120_dsv41_mixed.py`)**. Generated decode/prefill families for GB202. PTX histogram of the directory: `mma.sync.aligned.m16n8k16...bf16` (7752 occurrences), `m16n8k32...e4m3` (1568), `kind::mxf4nvf4...m16n8k64...e2m1` (686); staging is `cp.async.cg.shared::cta.global` (2323) with `cp.async.mbarrier.arrive.noinc`, i.e. non-TMA, with a "cp.async ring depth" knob (`csrc/cake_dsv4/README.md:120`, `README.md:187`). Launch bounds 256/320/384 threads. `fence.proxy.async.shared::cta` appears 44 times alongside `cp.async.bulk.global.shared::cta` (smem→global bulk stores). README headings say "SM120 / SM121" (`README.md:107`, `README.md:211`); source subdir is `sm_120a` (`cake_sparse_mla_sm120_dsv4_nvfp4.py:31`).
- **Hand-written sparse MLA SM120 (`include/flashinfer/attention/sparse_mla_sm120/`)**. Same three MMA families (`arch/mma_sm120.cuh`, `arch/mma_sm120_nvfp4.cuh`); staging helpers are 1-D `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` (not the `.tensor` form) plus `cp.async` (`arch/cp_async.cuh:63-89`). Split grid `dim3(num_tokens, h_blocks, allocated_splits)` and an LSE merge kernel (`execution/decode_launch.cuh:63-81`, `execution/merge.cuh:18-47`). Smem checked against `cudaDevAttrMaxSharedMemoryPerBlockOptin` (`execution/launch_validation.cuh:17`). SM121 gating: [ND].
- **`cake_vsa_sm120` (`csrc/cake_vsa_sm120/`, `flashinfer/cake_vsa_sm120.py`)**. Block-sparse attention, `HEAD_DIM = 128`, `BLOCK_SIZE = 64` (`cake_vsa_sm120.py:57-58`); persistent kernels with "two virtual CTAs per SM" (128-thread CTA ×2, or one 256-thread CTA) and split rows merged in a workspace (`cake_vsa_sm120.py:16-46`). MMA `m16n8k16` f16/bf16; staging `cp.async.bulk.tensor.4d...L2::cache_hint` (TMA) with `fence.proxy.async.shared::cta` (PTX histogram). Shipped for `sm_120a` and `sm_121a` (`flashinfer/jit/cake_vsa_sm120.py:15`, `:42`).
- **SM120 DSA indexer (`flashinfer/experimental/sm120_dsa_indexer/`)**. Lightning-indexer scores + exact top-k, ragged and paged, FP8 or MXFP4, 32/64 heads, head dim 128 (`README.md:3-16`). MMA `m16n8k32 e4m3` and `kind::mxf4nvf4...scale_vec::2X.m16n8k64 e2m1`; TMA `cp.async.bulk.tensor.2d/3d` with one `fence.proxy.async.shared::cta`; `kSmemLimit = 101376` (`csrc/sm120_dsa_indexer.cuh:74`). Persistent grids sized from the stream context's SM count (`README.md:26-27`). "on SM12x GPUs" (`README.md:5`).
- **DeepGEMM SM120 paged MQA logits (`flashinfer/experimental/deepgemm_sm120_paged_mqa_logits/`)**. Scheduler kernel + "persistent 384-thread register-MMA program (`mma.sync.m16n8k32 e4m3·e4m3→f32` + `ldmatrix` + TMA + mbarriers; no tcgen05/TMEM/WGMMA, which `ptxas` rejects on this architecture)" over 128-row KV segments (`README.md:15-20`). Stated target SM120a only (`README.md:10`).
- **`nvfp4_attention_sm120` (`csrc/nvfp4_attention_sm120/`, `include/flashinfer/attention/sm120/nvfp4_attention_sm120/`)**. Dense, non-causal, `Hq == Hkv`, head dim 128, BF16 in/out, on-device quantization (`flashinfer/experimental/nvfp4_attention/README.md:17-29`). MMA `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 e2m1`; has `primitives/tma.cuh` and `cp.async.bulk.commit_group/wait_group`. Decorated `@supported_compute_capability([120, 121])` (`flashinfer/nvfp4_attention_sm120.py:171`).
- **`cake_fp8_attention`, `cake_nvfp4_ptilemax_attention`, `cake_nvfp4_per_token`**: not SM12x kernels. READMEs state SM90 (`cake_fp8_attention/README.md:17`), SM103 (`cake_nvfp4_ptilemax_attention/README.md:8-9`), and SM100/SM103 with tcgen05 (`cake_nvfp4_per_token/README.md:7`, `:20`).

---

## 6. Shared infrastructure

- **SMEM capacity map keyed by arch string**: `get_smem_capacity_in_bytes("sm_120")` is called at `flashinfer/experimental/b12x/attention/varlen/_tuning.py:132` and `attention/_shared/contiguous/forward.py:391`, but the function and its `SMEM_CAPACITY_MAP` are imported from the external `cutlass` package (`flashinfer/attention/batch_invariant/swaps/operands.py:151-153`); the map itself is **not in this tree** [ND]. In-tree literals: 101376 at `b12x/attention/_shared/mla/smem.py:66`, `flashinfer/experimental/sm12x_mega_gate/heuristics.py:47`, `sm120_dsa_indexer/csrc/sm120_dsa_indexer.cuh:74`; `99u << 10` at `csrc/xqa/utils.cuh:51`.
- **Compile targets**: SM 12.0 → `compute_120f` on CUDA ≥ 12.9, `compute_120a` on 12.8; SM 12.1 → `compute_121a`, requires CUDA ≥ 12.9 (`flashinfer/compilation_context.py:96-107`).
- **Experimental policy**: env var `FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS` (`flashinfer/api_logging.py:2600`, `:2627`) gates only *automatic* selection; explicit `backend="<name>"` always works and warns once (`flashinfer/experimental/__init__.py:26-30`). `flashinfer/experimental/AGENTS.md:24` says experimental features are "JIT-only and must not ship in pre-built packages"; `AGENTS.md:25-29` forbids `backend="auto"` from picking experimental backends without the gate.
- **b12x status statements**: `flashinfer/experimental/README.md` and `AGENTS.md` contain no occurrence of the string "b12x" (grep) [SRC by absence]; b12x has no README of its own. Its status is carried by per-op `META` (`paged/__init__.py:37-85`) and the package docstring (`b12x/_api.py:1-17`).

---

## 7. Summary table

| Kernel | File | MMA family | K/V staging | Split-KV | Paged | FP8 KV | SM121 admitted |
|---|---|---|---|---|---|---|---|
| b12x paged decode | `b12x/attention/paged/forward_paged.py` | `mma.sync m16n8k16` bf16/f16 (`intrinsics.py:3957`); opt-in `mxf8f6f4 m16n8k32` (`:4083`) | TMA tiled atom, page id as coordinate (`:4107-4147`) | Yes, CUDA-graph policy (`planner.py:2024-2033`) | Yes, 64/128 (`planner.py:1945`) | Yes, dequant default (`:1685`) | Yes (`gating.py:31`) |
| b12x paged extend | `.../forward_extend_generic.py` | same | `cp.async` 16 B (`:3860-3870`); TMA one shape (`:3275-3290`) | No (`planner.py:2038`) | Yes | Yes | Yes |
| b12x merge | `.../merge.py` | none | direct global loads (`:11-13`) | reducer | n/a | n/a | Yes |
| FA2 prefill / TC decode | `include/flashinfer/attention/prefill.cuh` | `mma_sync_m16n16k16` (`:1706`) | `cp.async` (`:728`) | Yes (`scheduler.cuh:101`) | Yes | Yes (`:5046-5052`) | Yes (`utils.py:729`) |
| FA2 decode (no TC) | `.../decode.cuh` | none | `cp.async` (`:292`) | Yes (`scheduler.cuh:183-205`) | Yes | not determined | Yes |
| XQA MHA | `csrc/xqa/mha.cu` | `m16n8k16` f16/bf16 (`mma.cuh:36-44`) | `cp.async` (`ldgsts.cuh:37`) | Yes (`mha.cu:3123`) | Yes, 16–128 | Yes, dequant (`utils.cuh:752`) | Yes (`utils.cuh:50`) |
| XQA MLA SM120 | `csrc/xqa/mla_sm120.cu` | not determined | TMA (`:132`) | Yes (`:499`) | Yes | Yes (`jit/xqa.py:193`) | not determined |
| FMHAv2 | `csrc/fmha_v2/` | "HMMA" per comment (`generate_kernels.py:47`) | not determined | not determined | optional (`prefill.py:10129`) | ragged: no (`prefill.py:10446`); fixed-shape: yes | Yes (`prefill.py:9941`) |
| Cake DSv4 sparse MLA | `csrc/cake_dsv4/sm_120a/` | bf16 k16, e4m3 k32, mxf4nvf4 k64 | `cp.async` ring | not determined | Yes | mixed | README says yes (`README.md:107`) |
| Sparse MLA SM120 | `include/.../sparse_mla_sm120/` | same three | 1-D `cp.async.bulk` + `cp.async` (`arch/cp_async.cuh:69`) | Yes (`decode_launch.cuh:63`) | Yes | Yes | not determined |
| `cake_vsa_sm120` | `csrc/cake_vsa_sm120/` | `m16n8k16` f16/bf16 | TMA 4-D | row split (`cake_vsa_sm120.py:27`) | No (BSR blocks) | No | Yes (`jit/cake_vsa_sm120.py:42`) |
| SM120 DSA indexer | `experimental/sm120_dsa_indexer/csrc/` | e4m3 k32; mxf4nvf4 k64 | TMA 2-D/3-D | n/a (scores) | both forms | Yes | "SM12x" (`README.md:5`) |
| DeepGEMM paged MQA | `experimental/deepgemm_sm120_paged_mqa_logits/` | e4m3 k32 (`README.md:18`) | TMA (`README.md:18`) | KV segments | Yes | Yes | not determined (README: SM120a) |
| `nvfp4_attention_sm120` | `csrc/nvfp4_attention_sm120/` | mxf4nvf4 k64 e2m1 | TMA header present; detail not determined | not determined | No | No (NVFP4) | Yes (`nvfp4_attention_sm120.py:171`) |

---

## 8. Assumptions, unknowns, and discriminating checks

- Assumption: PTX-instruction histograms (section 5) count textual occurrences in generated sources, not executed paths.
- Unknown: the actual `shared_memory_per_multiprocessor` values on SM120 and SM121, hence real CTA residency for each trait.
- Unknown: whether `get_b12x_paged_attention` exists in vLLM, and the exact text of issue #6977.
- Smallest falsifying check for the `head_dim` 64 claim: on an SM120 device call `select_paged_forward_traits(cta_tile_q=16, head_dim_qk=64, head_dim_vo=64, q_dtype=torch.bfloat16, kv_dtype=torch.bfloat16)`; the claim predicts `ValueError: no valid NUM_MMA_KV typed shared-memory allocation fits the device`, and predicts success when `exact_num_mma_kv=2` is passed.
- Smallest falsifying check for the TMA-staging claim: disassemble the compiled decode kernel and grep for `cp.async.bulk.tensor`; the claim predicts it is present for decode and absent for an extend kernel with `cta_tile_q=64`.

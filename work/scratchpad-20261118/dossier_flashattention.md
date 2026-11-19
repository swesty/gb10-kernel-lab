# Dossier: FlashAttention kernels on SM120 / SM121 (workstation Blackwell)

Sources read on 2026-11-17:

- **Upstream** = `Dao-AILab/flash-attention` main at `9f9742c` (2026-11-17). Paths below with no prefix are relative to this repository.
- **Fork** = `vllm-project/flash-attention` at `8dd9eaea` (2026-11-13). Paths are prefixed `fork:`.
- Pull requests were read with `gh pr view` / `gh api` against `Dao-AILab/flash-attention`. PR text is the PR author's claim, not something measured for this dossier.

Labels used throughout: **[SRC]** = a fact read directly from source at the cited line. **[PR]** = a statement quoted or paraphrased from a pull-request body or comment. **[INF]** = an inference or arithmetic performed for this dossier. No repository code was executed; nothing in this dossier is a measurement made by the dossier author.

---

## Headline conclusions

1. **[SRC]** On upstream main, the FlashAttention-4 (CuTe DSL) forward kernel for compute capability 12.x is a 61-line subclass of the Ampere (SM80) kernel. The subclass uses warp-level `mma.sync` m16n8k16, `cp.async` loads, one K/V stage, 128 threads, and no TMA. The subclass overrides only `can_implement()` and forces `self.arch = Arch.sm_80`.
2. **[SRC]** On upstream main, the SM120 forward path rejects paged KV, block sparsity, `num_splits > 1`, FP8 inputs, and learnable sink. Dense, varlen, causal, local, and packed-GQA forward are wired.
3. **[SRC]** FlashAttention-2 (CUDA C++) has no SM120-specific branch. A capability-12.x device fails the `is_sm8x` predicate and receives the tiles tuned for A100/H100 (128x64, 4 warps for head dim 128).
4. **[SRC]** The vLLM fork at `8dd9eaea` contains `flash_attn/cute/flash_fwd_sm120.py`, but the fork's `is_fa_version_supported(4)` returns `False` for any device whose capability major is not exactly 10, and the fork's `flash_attn_varlen_func` has no `fa_version == 4` branch at all.
5. **[PR]** Two competing open lines of work would change conclusion 1: extending the SM80 subclass (#2336, #2348, #2980, #3060, #2998) versus replacing the subclass with SM12x-owned TMA kernels (#3108, #3109, #3132). None is merged.

---

## Part A — FlashAttention-4 (CuTe DSL), upstream main

### A1. `flash_fwd_sm120.py` and `flash_bwd_sm120.py`

**Class structure. [SRC]** `FlashAttentionForwardSm120(FlashAttentionForwardSm80)` (`flash_attn/cute/flash_fwd_sm120.py:15`); `FlashAttentionForwardSm80(FlashAttentionForwardBase)` (`flash_attn/cute/flash_fwd.py:579`). The file header states the design intent:

```python
# flash_attn/cute/flash_fwd_sm120.py:4-19
# SM120 uses the same SM80-era MMA instructions (mma.sync.aligned.m16n8k16) but has
# a smaller shared memory capacity (99 KB vs 163 KB on SM80). This module subclasses
# FlashAttentionForwardSm80 and overrides the SMEM capacity check accordingly.
...
class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    def __init__(self, *args, **kwargs):
        """Force SM80 code paths while the DSL still targets the resident SM120 GPU."""
        super().__init__(*args, **kwargs)
        self.arch = Arch.sm_80
```

**Why the arch override matters. [SRC]** `FlashAttentionForwardSm80.__call__` sets `self.use_tma_O = Arch.sm_90 <= self.arch < Arch.sm_120` (`flash_fwd.py:660`), and the epilogue's non-TMA branch stores O through shared memory with a vectorized global copy (`flash_fwd.py:418-431`). **[PR]** PR #2336's author reported that without the `__init__` override, `self.arch` was `Arch.sm_121a` on a DGX Spark and the kernel crashed with `tma_atom_O=None`.

**MMA atom. [SRC]** Both QK and PV use the 16-bit warp MMA:

```python
# flash_attn/cute/flash_fwd.py:588-599
def _get_tiled_mma(self):
    tiled_mma_qk = cute.make_tiled_mma(
        warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
        (self.num_threads // 32, 1, 1),
        permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
    )
    tiled_mma_pv = cute.make_tiled_mma(
        warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
        (self.num_threads // 32, 1, 1),
        permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
    )
```

**Threads, stages, register choice. [SRC]** The interface hard-codes these for SM120; the values are not read from the config object:

```python
# flash_attn/cute/interface.py:1484-1502
elif arch // 10 == 12:
    # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
    fa_fwd = FlashAttentionForwardSm120(
        dtype, head_dim, head_dim_v, qhead_per_kvhead,
        is_causal=causal, is_local=local, pack_gqa=config.pack_gqa,
        tile_m=config.tile_m, tile_n=config.tile_n,
        num_stages=1, num_threads=128, Q_in_regs=False,
        score_mod=score_mod, mask_mod=mask_mod,
        has_aux_tensors=aux_tensors is not None,
    )
```

128 threads = 4 warps, stacked along M (atom layout `(4, 1, 1)`). `Q_in_regs=False` means Q stays in shared memory (`SharedStorageQKV` with separate `sV`, `sQ`, `sK`, each 1024-byte aligned, `flash_fwd.py:601-620`). P is kept in registers: `rP` is built from `acc_S` and fed to `sm80_utils.gemm_rs` (`flash_fwd.py:1189-1205`); `sP_layout_atom = None` (`flash_fwd.py:585`). The O accumulator is written to shared memory by reusing the Q buffer (`# reuse sQ's data iterator`, `flash_fwd.py:1079-1080`).

**Tile sizes. [SRC]**

```python
# flash_attn/cute/config.py:644-666
tile_m, tile_n, mma_pv_is_rs, intra_wg_overlap = 128, 128, False, False
match inputs.device_capacity:
    case 8:
        tile_n = 64
    ...
    case 12 if inputs.head_dim > 64:
        # SM120 has 99 KB SMEM: 128×128 is preferred for D<=64, and 128×64
        # avoids the occupancy loss from a 96 KB tile for larger head dimensions.
        tile_n = 64
```

So SM120 forward gets 128x128 for head dim <= 64 and 128x64 for head dim > 64. A caller-supplied `tile_mn` overrides both (`config.py:668-671`).

**K/V loading. [SRC]** Loading is `cp.async` with a 128-bit copy atom, not TMA: `cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL)` with `num_bits_per_copy=128` (`flash_fwd.py:246-253`). `flash_fwd.py` imports no `flash_attn.cute.pipeline`; the import of `quack.copy_utils` (`flash_fwd.py:23`) is used only in the TMA-O epilogue branch that SM120 does not take. With `num_stages=1` the schedule inside `compute_one_n_block` is: wait for K, issue the V load, run QK MMA, wait for V and issue the next K load, softmax, run PV MMA (`flash_fwd.py:1125-1205`). **[INF]** With one stage, the loop overlaps the V load with the QK MMA and the next K load with softmax plus the PV MMA; there is no multi-block lookahead.

**Shared-memory budget. [SRC]**

```python
# flash_attn/cute/flash_fwd_sm120.py:47-58
smem_usage_Q = tile_m * head_dim * 2
smem_usage_K = tile_n * head_dim * num_stages * 2
smem_usage_V = tile_n * head_dim_v * num_stages * 2
smem_usage_QV = (
    (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
)
smem_usage = smem_usage_QV + smem_usage_K
# SM120 has 99 KB shared memory (vs 163 KB on SM80)
smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
if smem_usage > smem_capacity:
    return False
```

The numeric capacity lives in the `cutlass` Python package, not in this repository: **not found** in the tree. **[PR]** PR #2998 states the limit as 101,376 B (= 99 x 1024).

**[INF]** Applying the formula with one stage and `Q_in_regs=False`:

| head dim | tile | Q + K + V bytes | fits 101,376 B? |
|---|---|---|---|
| 64 | 128x128 | 16,384 + 16,384 + 16,384 = 49,152 | yes |
| 128 | 128x64 | 32,768 + 16,384 + 16,384 = 65,536 | yes |
| 128 | 128x128 | 3 x 32,768 = 98,304 | yes, 96 KB |
| 256 | 128x64 | 65,536 + 32,768 + 32,768 = 131,072 | **no** |

**`can_implement()` conditions. [SRC]** dtype in `{Float16, BFloat16}`; `head_dim % 8 == 0`; `head_dim_v % 8 == 0`; `tile_n % 16 == 0`; `num_threads % 32 == 0`; the shared-memory check above; `(tile_m * 2) % num_threads == 0` (`flash_fwd_sm120.py:37-61`). **[SRC]** `can_implement` is not called for the forward kernel on main: the only `can_implement(` call sites in `flash_attn/cute/` are the combine kernel (`interface.py:4047`) and an MLA backward (`interface.py:3331`), and the forward dispatch carries `# TODO: check @can_implement` (`interface.py:1507`). **[INF]** Head dim 256 on SM120 therefore passes host validation and fails later at launch. **[PR]** PRs #2980, #2998 and #3109 each state that main cannot run D256 on SM120. PR #2998's body says `can_implement` rejects D256, which does not match the call sites found at `9f9742c`; that discrepancy is unresolved.

**Backward. [SRC]** `FlashAttentionBackwardSm120(FlashAttentionBackwardSm80)` overrides only `can_implement` (budget = Q + dO + K + V; `flash_bwd_sm120.py:42-55`) and has no arch override. The backward configuration is fixed:

```python
# flash_attn/cute/config.py:3049-3064
def sm120_bwd_fields(inputs: BwdHeuristicInputs) -> dict:
    """The one SM120 backward configuration: SM80 MMA, 99 KB SMEM, 128 threads (4 warps)."""
    stages = 2 if inputs.head_dim <= 64 else 1
    return {
        ...
        "tile_m": 64, "tile_n": 64,
        "num_stages_Q": stages, "num_stages_dO": stages,
        "AtomLayoutMSdP": 4, "AtomLayoutNdKV": 4, "AtomLayoutMdQ": 4,
        "num_threads": 128, "spt": False,
    }
```

**Design-difference comments versus SM90/SM100.** Apart from the two header comments quoted above and the `config.py:664-665` occupancy comment, **not found**.

### A2. `interface.py`: architecture dispatch and SM120 restrictions

**Dispatch. [SRC]** `_get_device_arch()` returns `major * 10 + minor` from `torch.cuda.get_device_capability()`, overridable by `FLASH_ATTENTION_ARCH` (`interface.py:113-130`). The forward asserts `arch // 10 in [8, 9, 10, 11, 12]` (`interface.py:754`) and then branches: `arch // 10 == 8` -> `FlashAttentionForwardSm80` (`:1346`); `== 9` -> `FlashAttentionForwardSm90` (`:1364`); `in [10, 11]` -> `FlashAttentionForwardSm100`, the SwapAB decode class, or an MLA class (`:1387-1483`); `== 12` -> `FlashAttentionForwardSm120` (`:1484`). SM121 (GB10) maps to the same `12` bucket because of the integer division. The backward asserts `arch // 10 in [9, 10, 11, 12]` (`interface.py:2087`), so SM120 has an FA4 backward while SM80 does not.

**Every restriction found that applies when `arch // 10 == 12`. [SRC]**

| Feature | Restriction | Citation |
|---|---|---|
| `num_splits` | `raise ValueError("SM120 forward only supports num_splits=1")` when the requested value is not `None`/1 | `config.py:1542-1543` |
| Automatic split | `elif not is_sm100_family ...: # Only SM100/SM110 implement SplitKV ... num_splits = 1` | `config.py:1558-1560` |
| Paged KV, block sparsity | `case 12 if inputs.page_size is not None or inputs.use_block_sparsity: raise ValueError("SM120 forward does not support paged KV or block sparsity")` | `config.py:1887-1888` |
| FP8 | `assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."` | `interface.py:825` |
| MXFP8 QK | `raise NotImplementedError("QK MXFP8 requires SM100-family hardware")` | `interface.py:768-769` |
| `qv` (MLA) | `assert arch // 10 in [10, 11], "only support Blackwell arch with qv"` | `interface.py:866` |
| Learnable sink (forward) | `assert learnable_sink is None, "Learnable sink is not supported in this kernel"` inside the SM80 `__call__` | `flash_fwd.py:651` |
| Learnable sink (backward) | `assert arch // 10 in [9, 10, 11], "Learnable sink backward is supported on SM90 and SM100/SM110"` | `interface.py:2096` |
| Head dims | `_validate_head_dims` is skipped: `if arch // 10 not in [8, 12]:` | `interface.py:757-758` |
| SM100-only options | `raise ValueError(f"SM{capacity} forward does not support SM100-only options: ...")` | `config.py:1889-1895` |
| Backward | four asserts: block sparsity, `score_mod`, `mask_mod`, `deterministic` "not supported on SM 12.0" | `interface.py:2119-2123` |

**Not restricted on SM120 forward. [SRC]** `pack_gqa`: `default_pack_gqa` returns `False` only for `qhead_per_kvhead == 1` or an SM100-only hd256 condition (`config.py:684-709`), so GQA calls on SM120 default to packed and `config.pack_gqa` is passed through (`interface.py:1493`). Local window: `is_local=local` is passed (`interface.py:1492`) and the mask is applied with `mask_local=self.is_local` (`flash_fwd.py:1026-1027, 1046`); a stray `# TODO: local` remains at `flash_fwd.py:1070`. Varlen: the SM80 kernel selects `SingleTileVarlenScheduler` when `mCuSeqlensQ` or `mSeqUsedQ` is present (`flash_fwd.py:685-688`). Custom `score_mod` forward raises only for `arch // 10 == 8` (`interface.py:925-926`).

### A3. `paged_kv.py`

**Two mechanisms. [SRC]** Paged KV on SM90 and SM100 has a TMA path and a `cp.async` path, selected by whether the page size equals the KV tile: `paged_kv_non_tma=page_size not in [None, config.tile_n]` (`interface.py:1384`, `:1476`).

- **TMA path (page size == `tile_n`).** One page is exactly one KV tile, so the kernel indexes the page dimension of the TMA tensor with the page id (`flash_fwd_sm90.py:777-783`; `flash_fwd_sm100.py:3775`: "Currently we assume that page_size == n_block_size so we index into tXgX with block = 0").
- **`cp.async` path (`PagedKVManager`).** Each thread resolves `(page, offset)` for its rows through `resolve_paged_token` with a `FastDivmodDivisorV2`, computes a 64-bit global pointer per row, broadcasts the pointer across the threads that share a row with `utils.shuffle_sync`, and issues predicated 128-bit `cp.async` copies:

```python
# flash_attn/cute/paged_kv.py:90-108
# SM100 transposes V in gmem to (dv, page_size, num_pages);
# SM90 keeps V as (page_size, dv, num_pages), same layout as K.
v_gmem_transposed = arch != 90
universal_copy_bits = 128
async_copy_elems = universal_copy_bits // dtype.width
...
atom_async_copy = cute.make_copy_atom(
    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
    dtype, num_bits_per_copy=universal_copy_bits,
)
```

**Page-size constraints. [SRC]** `page_table` must be int32, contiguous in the last dim, and incompatible with `cu_seqlens_k` (`interface.py:650-655`); `page_table.shape[1] * page_size (+ window_size_right)` must be below 2**31 (`interface.py:844-847`); the non-TMA path requires head dims that are multiples of 16 (`config.py:1930-1931`; `flash_fwd_sm90.py:70-72`); SM100 2CTA requires `page_size in (None, 128)` (`config.py:782`); the SwapAB decode kernel requires `page_size % 128 == 0` (`config.py:1143, 1251-1253`).

**Why SM120 is excluded on main. [SRC]** The SM80 `__call__` accepts `mPageTable` (`flash_fwd.py:635`) and `flash_fwd.py` contains no other reference to that name, so the kernel would silently ignore a page table; `config.py:1883-1888` rejects the call instead. `PagedKVManager.load_KV` has only two shared-memory layout branches, `self.arch == 90` and the SM100 flatten (`paged_kv.py:243-249`). **[INF]** The exclusion is "not implemented", not a hardware limit: the manager is `cp.async`-based and SM120 supports `cp.async`. **[PR]** PR #3060 makes exactly that argument ("With a few lines of patch, we can extend Hopper's support to cover Ampere and RTX Blackwell").

### A4. SplitKV on other architectures

**Heuristic. [SRC]**

```python
# flash_attn/cute/config.py:197-214
def num_splits_heuristic(total_mblocks: int, num_sms: int, num_n_blocks: int, max_splits: int) -> int:
    # If num_n_blocks is too small, use 1 split. ...
    if num_n_blocks <= 4:
        return 1
    if total_mblocks == 0:
        return 1
    ...
    return max(1, min(num_sms // total_mblocks, max_splits, num_n_blocks))
```

Inputs (`config.py:1552-1562`): `total_mblocks = batch_size * num_heads_kv * heads_per_m_block * num_m_blocks` (where `heads_per_m_block` is 1 under packed GQA), `num_sms` from `get_num_sms_for_selection` (the device SM count or `FLASH_ATTENTION_NUM_SMS`; `cute_dsl_utils.py:52-67`), `num_n_blocks = ceil(seqlen_k_loaded / tile_n)`, and `max_splits = 128`. The heuristic is reached only for `device_capacity in (10, 11)`; SM90 FA4 also gets `num_splits = 1`.

**Combine kernel. [SRC]** `FlashAttentionForwardCombine` defaults to `tile_m=8, k_block_size=64, log_max_splits=4, num_threads=256, stages=4` and loads fp32 partials with `cp.async` (`flash_fwd_combine.py:28-40, 107-111`). Partial O and LSE are allocated as fp32 with a leading `num_splits` dimension (`interface.py:910, 918`).

**`flash_fwd_decode_sm100.py`. [SRC]** A separate "SwapAB" single-token decode kernel that computes `S^T = K Q^T` and `O^T = V^T P^T`, "so KV tokens fill the 128-row MMA M dimension and the query heads of one KV head fill the small N dimension"; one CTA is 16 warps (`flash_fwd_decode_sm100.py:1-11`). The SwapAB policy uses a 256-token KV tile and a different split rule aimed at a target CTA count (64 by default, 128 for head dim 64) rather than at SM occupancy: `pow2_splits = 1 << (max(1, target // bh).bit_length() - 1)` capped by `max_splits=32` and `seqlen_k // 512` (`config.py:1142-1158, 1299-1310`). The policy admits only SM100 and SM103 (`config.py:1242`).

### A5. Contrast: SM90 and SM100 forward

**`flash_fwd_sm90.py` (Hopper). [SRC unless marked]** `FlashAttentionForwardSm90` derives from the same `FlashAttentionForwardBase` (`:54`). Both MMAs are warp-group `wgmma` built with `sm90_utils_basic.make_trivial_tiled_mma`, with a 64-row atom and `atom_layout_mnk=(tile_m // 64, 1, 1)` (`:100-122`). Thread count is `128 * (num_wg_mma + 1)`: one producer warp group plus one to three MMA warp groups (`:216-220`). Q, K and V are staged by TMA through `PipelineTmaAsync` pipelines guarded by mbarriers (`:493-521`), with only 32 producer threads issuing TMA (`:221`) and `setmaxregister_decrease/increase` moving registers from the producer to the MMA groups (`:615, 640`). The interface passes `num_stages=2` for K/V (`interface.py:1376`). P is consumed from registers when `mma_pv_is_rs` is true and from shared memory otherwise (`:89-97, 118-120`). For head dim 128 the tile is 128x128 with register-source PV and intra-warpgroup overlap (`config.py:272-273`). **[INF]** Head dim 128 therefore runs 2 MMA warp groups + 1 producer group = 384 threads. The SM90 shared-memory constant is `SM90_SMEM_BYTES = 227 * 1024` (`config.py:3179`).

**`flash_fwd_sm100.py` (datacenter Blackwell). [SRC unless marked]** `FlashAttentionForwardSm100` is a standalone class (`:173`) using `tcgen05` MMA with accumulators in tensor memory (TMEM): S occupies columns `[0, n_block_size]` and O follows (`:432-438`), and the config rejects layouts where `2 * tile_n + q_stage * head_dim_v_padded > 512` TMEM columns (`config.py:1914-1915`). One CTA is 512 threads with role-specialized warps: softmax warps 0-3 and 4-7, correction warps 8-11, a single MMA warp 12, load warp 14, empty warp 15 (`:363-384`). K/V arrive by TMA with `kv_stage` chosen to fill a 224 KB budget (`config.py:970, 1076`; `2 <= kv_stage`, `:312`). `q_stage = 2` (two Q tiles per CTA) when the packed query length exceeds `tile_m` and `head_dim_v != 256` (`config.py:1549`). The 2-CTA mode sets `cluster_shape_mn = (2, 1)` and `tcgen05.CtaGroup.TWO`, making the MMA tile M equal to `2 * m_block_size` (`:239-252, 550`). For head dim 128 the default tile is 128x128 (`config.py:644`), and 2-CTA is eligible only for non-causal, non-local, unsplit calls (`config.py:775-793`). Softmax warps signal the MMA warp partway through P (`split_P_arrive`, `:220-226`). **[INF]** Neither `wgmma`, `tcgen05`, nor TMEM exists on SM120, which is why upstream reused the SM80 kernel rather than either of these.

### A6. Pull requests

| PR | State | Size | Author |
|---|---|---|---|
| #2329 SM120 forward pass | MERGED 2026-03-12 | +102/-7, 3 files | blake-snc |
| #2336 SM120 split-KV | OPEN | +262/-178, 3 files | blake-snc |
| #2348 SM120 paged KV | OPEN | +561/-224, 4 files | blake-snc |
| #2980 E4M3 forward on SM120 | OPEN | +1885/-641, 10 files | joeyearsley |
| #2998 SM120 BF16/FP16 tile tuning | OPEN | +17/-0, 1 file | joeyearsley |
| #3026 FA2 SM120 hdim128 tiles | OPEN | +3/-2, 1 file | athrva98 |
| #3060 paged KV for SM80/SM120 | OPEN | +310/-40, 7 files | xrq-phys |
| #3108 building blocks for #3109 | OPEN | +342/-18, 9 files | lucifer1004 |
| #3109 SM12x-owned forward + backward | OPEN | +16216/-290, 31 files | lucifer1004 |
| #3132 FP8 forward on SM12x | OPEN, draft | +17412/-320, 34 files (includes #3109) | lucifer1004 |

All open PRs show `reviewDecision = REVIEW_REQUIRED`.

**#2329 (merged). [PR]** Added the `FlashAttentionForwardSm120` subclass, the `interface.py` dispatch, and a fix for an SM80 causal bug (`is_first_n_block=True` was passed on every causal iteration, resetting the online-softmax running max). The tile table in the PR body gives 48 KB for 128x128 at D<=64 and 64 KB for 128x64 at D>64, noting "128×128 would use 96 KB". Listed limitations: no SplitKV, paged KV, varlen, backward, or TMA. **Benchmark numbers: none**; the body reports only correctness, 17/17 configurations on "NVIDIA GB10 (DGX Spark, SM121a)" with `max_diff < 0.016`.

**#2336. [PR]** Adds `num_splits > 1` to the SM80/SM120 kernel by threading `is_split_kv` through the base class and writing fp32 partial O directly from MMA accumulator registers to global memory, bypassing shared memory. Validation: 14/14 split configurations on SM121a. No timing numbers. **Reviewer blocker (tridao, 2026-03-13):** "The paged KV stuff doesnt' look right, we should not be indexing in pytorch, that will kill perf ... For split we should be writing in fp32, not bf16." The author split the PR and reported both points addressed; no maintainer response after that was found.

**#2348. [PR]** In-kernel paged KV for SM120 using `PagedKVManager`'s copy atom, with plain Int32 division instead of `FastDivmodDivisor`. A commenter measured paged KV at 10x the dense kernel time; tridao replied "I don't think perf diff should be 10x. In the past we've see 0-10% perf diff between dense and paged KV." The author then moved to `num_stages=2` and fixed a floor-division bug (`n_block_size // num_threads = 64 // 128 = 0`). Reported numbers on **SM121a (DGX Spark), BF16, page_size=64, Sq=1**: B=1 Sk=2048 H=16 D=128 dense 0.080 ms vs paged 0.095 ms (1.18x); B=4 Sk=4096 H=16 D=128 0.577 vs 0.652 ms (1.13x); B=4 Sk=2048 H=16 D=64 0.149 vs 0.163 ms (1.09x). The PR is stacked on #2336.

**#2980. [PR]** Adds an E4M3 `mma.sync.m16n8k32` path to the SM80-derived SM120 kernel with BF16 output, plus SplitKV (BF16 and FP8) and FP8-only paged KV with any positive page size; BF16/FP16 paged KV is rejected explicitly. V is read with `ldmatrix.m16n16.trans.b8` with no repack; default FP8 tiles are 64x64. Numbers, all on **RTX PRO 6000 Blackwell (SM120)**, CUDA-graph replay: FP8 prefill geomean **1.31x vs FA2 BF16** (49 rows, 49/49 wins) and **1.40x vs upstream FA4 BF16** (47 rows); decode (216 shapes, automatic splits) FP8 **1.70x vs FA2** (103/108 faster, worst 0.85x) and BF16 **1.17x vs FA2** (61/108 faster, worst 0.43x at b32, K=8192, MQA, D64); paged vs contiguous FP8 geomean **0.883x** (worst 0.71x). Third-party validation comments from xzwgit on three SM120 nodes. No maintainer review found.

**#2998. [PR]** A 17-line change to SM120 forward tile selection: fall back to 64x64 when the default does not fit (fixing D256), then 64x128 for narrow causal heads, 128x64 for narrow non-causal heads, 64x64 for wider heads. **RTX PRO 6000 Blackwell**, 49 shapes at a held 1400 MHz: **1.19x geomean vs default FA4** (46/47 faster, range 0.997x-1.87x) and **1.11x vs FA2** (45/49 faster). No comments.

**#3026. [PR]** Adds capability 12.0 to FA2's `is_sm8x` predicate in `run_mha_fwd_hdim128` so SM120 gets 128x32 non-causal and 64x64 causal; capability 12.1 (GB10) is explicitly not covered. Rationale in the body: "64KB per block means only 1 CTA per SM on `sm_120`, 4 of 48 warps"; ncu theoretical occupancy 8.3% -> 16.7%. Numbers: **RTX 5060 Laptop** 1.059x-1.083x; **RTX PRO 6000 Blackwell Server Edition (188 SMs)** 1.024x-1.086x non-causal and about 1.00x causal. No maintainer response.

**#3060. [PR]** Body is one sentence (quoted in A3). Touches `paged_kv.py` (+32/-22), `flash_fwd.py` (+49/-13), `config.py` (+5/-4). Review comments from bvolpato on an **RTX 5070 Ti** reported two correctness bugs: wrong output channels 24-31 when `D=24, Dv=32` alias the V predicate to K's, and an "Invalid `__shared__` write of size 16 bytes" for `D=24, Dv=40, tile_mn=(64, 96)`. The author acknowledged both. No benchmark numbers.

**#3108. [PR]** Architecture-independent groundwork: SM count taken from the current stream's context (green contexts), a backward `mask_mod` window fix, opt-in `AttentionMask` switches, and per-batch SplitKV pieces in the prepare and combine kernels. Tested only on RTX PRO 6000 Blackwell. No comments.

**#3109. [PR]** Replaces both SM80 subclasses with kernels under `flash_attn/cute/sm120/`: a warp-MMA forward **with TMA K/V**, head dims up to 256, paged KV, block sparsity, SplitKV including per-batch splits, a transposed paged-decode kernel, and a new backward. The author states the forward is a port of the SGLang SM12x FA4 implementation. Numbers versus `main` at `4616f511`: **RTX PRO 6000 Blackwell (188 SMs)** training geomean over 111 cases forward **1.34x**, backward **1.15x**, combined **1.21x**; versus FA2 2.8.4 forward 1.13x; decode on a dense cache **2.54x vs main** (up to 12.3x), 1.15x vs FA2 dense, 1.12x vs FA2 paged. **GB10 (48 SMs)**: hd128 GQA S8192 causal forward 1.16x; hd128 GQA S8192 local 4.17x; decode with one 32K request 1.7-2.0x. **RTX PRO 5000 (110 SMs)**: same causal case 1.26x. Note: the body describes main's SM120 path as "fixed 64 x 64 tiles", which matches the backward (`config.py:3055-3056`) but not the forward (128x128 / 128x64). No reviews.

**#3132 (draft). [PR]** FP8 (e4m3 and e5m2) forward for the #3109 kernels using m16n8k32 PV with fp8 P scaled by 2**8. **RTX PRO 6000 Blackwell (188 SMs)**: fp8 vs bf16 prefill geomean **1.69x** (40 cases), paged decode **1.84x** (24 cases, page 256). The body states the PR is not stacked on #2980 and that #3109 replaces the file #2980 edits.

---

## Part B — FlashAttention-2 (CUDA C++)

### B7. Tiles and staging

**Kernel traits. [SRC]** The MMA atom is `SM80_16x8x16_F32F16F16F32_TN` or the BF16 variant for any `__CUDA_ARCH__ >= 800` (`csrc/flash_attn/src/kernel_traits.h:29-34`), tiled as `Layout<Shape<Int<kNWarps>,_1,_1>>` with `Tile<Int<16 * kNWarps>, _16, _16>` (`:74-77`). Global loads use `SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>` (`:125-129`); the main loop sequences loads with `cute::cp_async_fence()` and `cp_async_wait<0>()` (`csrc/flash_attn/src/flash_fwd_kernel.h:261-409`). There is no TMA anywhere in FA2.

**Per-head-dim tiles in `run_mha_fwd_hdim*`. [SRC]**

```cpp
// csrc/flash_attn/src/flash_fwd_launch_template.h:251-265
auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
bool is_sm8x = cc_major == 8 && cc_minor > 0;
...
    // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
    // and 128 x 32 (48 KB smem) is the fastest for non-causal since we get 2 CTAs per SM.
    if (is_sm8x) {
        if constexpr(!Is_causal) { /* 128, 32, 4 warps */ } else { /* 64, 64, 4 warps */ }
    } else {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, ...>(params, stream);
    }
```

| head dim | tile (kBlockM x kBlockN, warps) for a capability-12.x device | citation |
|---|---|---|
| 32, 64 | 128x128, 4 (no capability branch) | `:200, 212` |
| 96 | 128x64, 4 (`is_sm8x` false) | `:238` |
| 128 | 128x64, 4 (`is_sm8x` false) | `:264` |
| 192 | 128x64, 8 | `:288` |
| 256 | 64x64, 4 (chosen by a shared-memory test, below) | `:317-321` |

**What SM120 gets. [INF from SRC]** `cc_major == 8` is false for capability 12.0 and 12.1, so an SM120/SM121 device takes the non-`is_sm8x` branch, whose comment says "1st ones are good for H100, A100" (`:272`). No file under `csrc/flash_attn/src/` tests for capability 12.

**Split-KV tiles. [SRC]** `kBlockM = 64` and `kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64)`, 4 warps (`:183-186, 193`). An explicit `num_splits == 1` request routes to an "align" variant with `kBlockN = Headdim <= 64 ? 128 : 64` for bitwise parity with the standard kernel (`:174-179, 187-191`).

### B8. Split-KV and paged KV

```cpp
// csrc/flash_attn/flash_api.cpp:282-316 (abridged)
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    ...
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        ...
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
        ...
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) { return num_splits; }
    }
    return 1;
}
```

**Inputs. [SRC]** The call is `num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128)` with the comment "We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block" (`flash_api.cpp:335-336`). `num_m_blocks = ceil(max_seqlen_q / 64)` and `num_n_blocks = ceil(max_seqlen_k / block_n)` (`:324-328`). `num_sm` is `cudaDevAttrMultiProcessorCount` (`csrc/flash_attn/src/hardware_info.h:37-41`). The heuristic picks the smallest split count whose wave-fill efficiency is within 85% of the best. **[INF]** The `num_sm * 2` assumption (two CTAs per SM) is an occupancy assumption inherited from A100/H100; whether two 80 KB split-KV CTAs fit on one SM120 SM depends on per-SM shared memory, which is **not determinable from source**. **[PR]** PR #3026 states a 64 KB block yields 1 CTA per SM on `sm_120`.

**Kernels. [SRC]** `flash_fwd_splitkv_kernel` -> `compute_attn_splitkv` (`flash_fwd_launch_template.h:41-47`); when `Split` is true the output element type is fp32 (`ElementO = ... ElementAccum`, `flash_fwd_kernel.h:530`) and each split covers `n_blocks_per_split` blocks (`:537-541`). `flash_fwd_splitkv_combine_kernel` is instantiated for `Log_max_splits` 1..7 with `kBlockM = 4` when the head dim is divisible by 128 (`flash_fwd_launch_template.h:138-162`).

**Paged KV. [SRC]** Paged KV exists only in the split-KV kernel: `run_mha_fwd(params, stream, /*force_split_kernel=*/... || paged_KV)` (`flash_api.cpp:773, 1511`). The kernel re-bases the K and V global pointers per block from `block_table` (`flash_fwd_kernel.h:596-606, 775-785, 873-880`). Upstream constraint: `TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256")` (`flash_api.cpp:614, 1306`). In `varlen_fwd`, split is configured only for the single-query GQA "ngroups swapped" case; otherwise `TORCH_CHECK(num_splits <= 1, "num_splits > 1 is not supported for varlen paged KV")` (`flash_api.cpp:731-739`). Paged KV with `leftpad_k` is rejected (`:743`).

### B9. Shared memory

**[SRC]** `kSmemQSize = size(SmemLayoutQ) * sizeof(Element)`, `kSmemKVSize = size(SmemLayoutKV) * 2 * sizeof(Element)`, `kSmemSize = Share_Q_K_smem ? max(Q, KV) : Q + KV` (`kernel_traits.h:107-109`). The only opt-in call:

```cpp
// csrc/flash_attn/src/flash_fwd_launch_template.h:84-87 (repeated at :125-128)
if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
}
```

Head dim 256 is the only place FA2 queries the device: `cudaDevAttrMaxSharedMemoryPerMultiprocessor` and `cudaDevAttrMaxSharedMemoryPerBlockOptin`, selecting 128x64 with 8 warps only if `max_smem_per_block >= 2 * Headdim * (128 + 2 * 64)` (`:303-321`).

**[INF]** Byte counts for a 2-byte element: hd128 standard 128x64 = 32,768 + 32,768 = 65,536 B (64 KB); hd128 split-KV 64x128 = 16,384 + 65,536 = 81,920 B (80 KB); hd256: the 128x64 test needs 131,072 B, which exceeds 101,376 B, so SM120 takes 64x64 = 32,768 + 65,536 = 98,304 B (96 KB). All three fit under 101,376 B.

---

## Part C — the vLLM fork at `8dd9eaea`

**C10a. Does the fork contain `flash_attn/cute/flash_fwd_sm120.py`? [SRC]** Yes. `diff -rq` reports no difference for `flash_fwd_sm120.py`, `flash_bwd_sm120.py` or `paged_kv.py` between the fork and upstream `9f9742c`. The fork's `interface.py` dispatches `elif arch // 10 == 12:` at `fork:flash_attn/cute/interface.py:1687`.

**C10b. Version-support functions. [SRC]** Both are defined:

```python
# fork:vllm_flash_attn/flash_attn_interface.py:57-63
def _is_fa4_supported(device = None) -> Tuple[bool, Optional[str]]:
    if not FA4_AVAILABLE:
        return False, f"FA4 is unavaible due to: {FA4_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] != 10:
        return False, \
            "FA4 is only supported on devices with compute capability == 10"
    return True, None
```

`is_fa_version_supported(4, device)` returns element `[0]` and `fa_version_unsupported_reason(4, device)` returns element `[1]` (`:65-82`). On capability 12.x the functions return `False` and the string `"FA4 is only supported on devices with compute capability == 10"` (or the import-failure string if `flash_attn.cute.interface` failed to import). `DEFAULT_FA_VERSION = 2` (`:38`), and `_is_fa2_supported` requires only capability major >= 8 (`:40-46`), so FA2 is reported as supported on 12.x.

**[SRC]** In addition, the fork's `flash_attn_varlen_func` implements only `fa_version == 2` and `fa_version == 3`; any other value reaches `raise ValueError(f"Unsupported FA version: {fa_version}")` (`fork:vllm_flash_attn/flash_attn_interface.py:242-309`). **[INF]** FA4 is therefore not reachable through that wrapper on any architecture at this commit; how vLLM itself invokes FA4 on capability 10 was not examined.

**C10c. FA4 paged KV on SM120 in the fork. [SRC]** No. `fork:flash_attn/cute/config.py:1659` raises `"SM120 forward does not support paged KV or block sparsity"` and `:1355-1356` raises `"SM120 forward only supports num_splits=1"`.

**C10d. Fork FA2 differs from upstream FA2 in the paged path. [SRC]** The fork relaxes the page constraint to `page_block_size % 16 == 0` (`fork:csrc/flash_attn/flash_api.cpp:638, 1373`) using an added `GmemTiledCopyQKVPaged` copy that gives each thread a contiguous tile so "each thread need only fetch 1 page index per mainloop iteration" (`fork:csrc/flash_attn/src/kernel_traits.h:135-144`). The fork's tile dispatch in `flash_fwd_launch_template.h` differs from upstream only in error-check macros and one `is_even_MN` predicate, so the SM120 tile outcome in B7 is the same. The fork's wrapper raises `NotImplementedError("FA2 does not support num_splits > 1")` and defaults `num_splits=0` (`fork:vllm_flash_attn/flash_attn_interface.py:162, 251-252`); **[INF]** with `num_splits=0` the C++ heuristic runs only in the single-query GQA branch (`fork:csrc/flash_attn/flash_api.cpp:757-762`).

**C10e. Distance from upstream. [SRC]** Fork `git log -3 -- flash_attn/cute`: `5970047` 2026-11-13 "[Cute,Fwd,Sm90] Finish the FP8 QK-split scores before the PV passes (#218)"; `af5422c` 2026-11-05 "Sync Dao-AILab FlashAttention through 54b044c8 (#213)"; `9ed6b12` 2026-10-29 "Add SM90 FP8 MMA forward for head dimension 512 (#199)". The upstream clone is depth 1, so `git log -3` there shows only `9f9742c`. The GitHub compare API reports upstream `54b044c8` (2026-11-02) -> `9f9742c` as 32 commits ahead with 22 changed files under `flash_attn/cute/`. File lists: 26 files under `flash_attn/cute` differ; only in upstream: `blockscaled_utils.py`; only in the fork: `flash_fwd_mla_decode_sm100.py`, `hopper_helpers.py`, and three `sm100_hd256_2cta_fmha_backward*.py` files. **[INF]** The fork's cute directory is roughly two weeks and 32 upstream commits behind while carrying fork-only SM90 FP8 patches; none of the differences touch the SM120 files.

---

## Feature matrix

| Feature | FA2 (C++), SM120 device | FA4 SM90 | FA4 SM100 | FA4 SM120, main | FA4 SM120, open PRs |
|---|---|---|---|---|---|
| MMA instruction | `mma.sync` m16n8k16 | `wgmma` | `tcgen05` + TMEM | `mma.sync` m16n8k16 | same; m16n8k32 for FP8 (#2980, #3132) |
| K/V staging | `cp.async` | TMA, 2 stages | TMA, `kv_stage` >= 2 | `cp.async`, 1 stage | TMA K/V (#3109); 2 stages (#2348) |
| Threads per CTA | 128 (256 at hd192) | 384 at hd128 [INF] | 512 | 128 | 8 QK + 8 PV warps for one hd256 FP8 tile (#3132) |
| Tile at head dim 128 | 128x64; split-KV 64x128 | 128x128 | 128x128, optional 2-CTA | 128x64 | 64x64 (#2998); per-call model (#3109) |
| Head dim 256 | 64x64, 96 KB [INF] | 128x80 | yes | fails at launch [INF, PR] | #2998, #2980 (FP8), #3109 |
| Split-KV | yes, heuristic on `2 * num_sm` | no (forced 1) | yes, plus SwapAB decode | no (`ValueError`) | #2336, #2980, #3109 |
| Paged KV | yes; page % 256 upstream, % 16 in fork | TMA if page == tile_n, else `cp.async` | same, plus SwapAB (page % 128) | no (`ValueError`) | #2348, #3060 (BF16); #2980 (FP8 only); #3109 |
| FP8 forward | no | fork-only patches | yes | no (assert) | #2980 (E4M3), #3132 (E4M3 + E5M2) |
| Varlen / local / packed GQA | yes / yes / via ngroups swap | yes | yes | yes / yes / yes | unchanged or extended |
| Learnable sink | not found | yes | yes | no (assert) | #3109, #3132 |
| Block sparsity | fork-only sparse kernels | yes | yes | no | #3109 |
| Backward | yes | yes | yes | yes, fixed 64x64, hd <= 128 [INF] | #3109 |
| Reachable from the fork's vLLM wrapper on 12.x | yes (default) | n/a | n/a | no | no |

---

## Assumptions, counterexamples, and unresolved items

- The 99 KB figure is taken from source comments and from PR #2998's "101,376 B"; the value returned by `cutlass.utils.get_smem_capacity_in_bytes("sm_120")` was not read because the `cutlass` package is not in either tree.
- The claim that head dim 256 fails at launch on main is an inference from the byte arithmetic plus the absence of a `can_implement` call, corroborated by three PR bodies. PR #2998 attributes the failure to `can_implement`, which conflicts with the call sites found; the exact failure point is unverified.
- Per-SM shared memory and the resulting CTAs-per-SM on SM120 and SM121 are not in source. The only occupancy data are PR #3026's ncu figures for an RTX 5060 Laptop.
- All benchmark numbers are PR authors' claims. Only #2329, #2336, #2348, and parts of #3109 and #3132 report DGX Spark (SM121) results; the remaining numbers are RTX PRO 6000 Blackwell, RTX PRO 5000, RTX 5060 Laptop, or RTX 5070 Ti.
- No maintainer review exists on any open PR except tridao's comments on #2336 and #2348, so "reviewer blockers" for the others are **not found**, not "none".
- Whether the fork-only cute files are fork additions or files upstream removed after `54b044c8` was not determined.

## Smallest discriminating experiments

1. **FA4 SM120 smem claim.** On an SM120 or SM121 host, call `flash_attn.cute.interface.flash_attn_func` with BF16 inputs of head dim 256. A launch-time shared-memory error confirms "passes host validation, fails at launch"; a host-side `ValueError` or a successful run falsifies that inference.
2. **FA2 tile claim.** Run `ncu` (or `cuobjdump` on the launched kernel symbol) for an FA2 hd128 non-causal forward on the SM120 host and read the template arguments of `flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, ...>>`. A `128, 32` or `64, 64` instantiation falsifies conclusion 3.
3. **Fork gating claim.** On the SM12x host, `python -c "from vllm_flash_attn.flash_attn_interface import is_fa_version_supported as f, fa_version_unsupported_reason as r; print(f(4), r(4), f(2))"`; the expected output is `False`, the "compute capability == 10" string, `True`.

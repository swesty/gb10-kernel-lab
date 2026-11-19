# Dossier: instruction set and resource limits of workstation Blackwell (sm_120 / sm_121)

Research date 2026-11-17. Sources read directly:

- **PTX** = PTX ISA manual, version 9.5 (CUDA 13.5), https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section numbers given; the page is a single HTML document).
- **TUNE** = Blackwell Tuning Guide v13.5, https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- **PROG** = CUDA Programming Guide v13.5, section 5.1 Compute Capabilities, https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- **CUTLASS** = local clone `cutlass/` at commit b2f249a (2026-11-06). Citations are `cutlass/path:line`.

Labels used below: **[PTX]**, **[TUNE]**, **[PROG]**, **[CUTLASS]** mark text read verbatim from a primary source. **[INFERENCE]** marks a conclusion drawn by this dossier and not stated by a source. Nothing was compiled or executed, so every statement about what `ptxas` accepts rests on the manual's text, not on a build.

Files the brief named that do not exist in this clone: `include/cutlass/arch/mma_sm120.hpp` (the SM120 ops live only under `include/cute/arch/`), any `mma_sm120_desc*` file, and any symbol named `enable_sm120_family`. All three: **not found**.

---

## 1. MMA on SM120 / SM121

**Conclusion.** sm_120 and sm_121 have no warpgroup or tensor-memory matrix instruction. The only tensor-core instruction is the warp-level, register-operand `mma.sync` (and `mma.sp`), extended with FP8/FP6/FP4 `.kind::f8f6f4` and block-scaled `.kind::mxf8f6f4` / `.kind::mxf4` / `.kind::mxf4nvf4` forms.

### 1a. BF16 / FP16 `m16n8k16`

[PTX §9.7.16.5.14, Target ISA Notes] "`.f16` floating point type `mma` operation with `.m16n8k16` shape requires sm_80 or higher." and "`.bf16` alternate floating point type `mma` operation with `.m16n8k8` and `.m16n8k16` shapes requires sm_80 or higher."

[INFERENCE] Because the non-suffixed targets follow the onion model ([PTX §11.1.2] "each generation adds new features and retains all features of previous generations"), `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` is available on baseline `sm_120` and `sm_121` with no `a`/`f` suffix. CUTLASS agrees: `cutlass/include/cute/arch/mma_sm80.hpp:39` enables the SM80 ops with `#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))`, and the BF16 string is at `mma_sm80.hpp:239`.

Caveat: CUTLASS ships no BF16/FP16 SM120 GEMM collective. `cutlass/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:80` asserts "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA." A BF16 attention kernel on sm_120 therefore has no CUTLASS reference collective; the SM80 atom plus the SM120 TMA/pipeline scaffolding must be combined by hand.

### 1b. Native FP8

Two distinct FP8 encodings exist.

1. Legacy Ada form, no `.kind`: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. [PTX] "`.e4m3` and `.e5m2` alternate floating point type `mma` operation requires sm_89 or higher"; introduced in PTX ISA 8.4. [CUTLASS] `cutlass/include/cute/arch/mma_sm89.hpp:79`, guarded by `__CUDA_ARCH__ >= 890` (`mma_sm89.hpp:50`). [INFERENCE] Available on baseline sm_120/sm_121.
2. Blackwell form, `.kind::f8f6f4`: `mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype` with `.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1}` [PTX syntax block]. [PTX Target ISA Notes] "Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier requires sm_120a and are supported on sm_120f or higher in the same family from PTX ISA version 8.8." Introduced in PTX ISA 8.7.

So FP8 `mma.sync` is native on sm_120, in a family-wide way: the legacy form is baseline, the `.kind::f8f6f4` form needs `sm_120a` or `sm_120f` (and `sm_120f` code runs on 12.1, see section 5). PTX 8.7 also added "`.f16` type accumulator and shape `.m16n8k16` with FP8 types" [PTX §13.8]; the `mma.sp` notes say that variant "requires sm_120", the dense `mma` notes give no separate target line for it.

CUTLASS op (representative; 25 A/B type pairs × {f32, f16} accumulators):

```
cutlass/include/cute/arch/mma_sm120.hpp:68
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
```

Registers per thread: `ARegisters = uint32_t[4]`, `BRegisters = uint32_t[2]`, `C/DRegisters = float[4]` (`mma_sm120.hpp:57-60`). The traits inherit the SM80 int8 layout: `MMA_Traits<SM120_16x8x32_TN<...>> : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>` with comment "The MMA accepts 8-bit inputs regardless of the types for A and B" (`cutlass/include/cute/atom/mma_traits_sm120.hpp:121-125`).

### 1c. Block-scaled kinds

[PTX syntax]

- `.kind::mxf8f6f4`: `m16n8k32`, `.scale_vec::1X`, `.stype = .ue8m0`.
- `.kind::mxf4`: `m16n8k64`, `.e2m1` only, `.scale_vec::2X`, `.ue8m0`.
- `.kind::mxf4nvf4`: `m16n8k64`, `.e2m1` only, `{.scale_vec::2X, .scale_vec::4X}`, `{.ue8m0, .ue4m3}`. "Support for `.scale_vec::4X` on `.ue8m0` as `.stype` with `.kind::mxf4nvf4` is introduced in PTX ISA version 9.1."

Target requirements, dense vs sparse:

| Instruction | Feature | PTX text |
|---|---|---|
| `mma` (dense) | `.kind`, `.block_scale`, `.scale_vec_size` | "requires sm_120a and are supported on sm_120f or higher in the same family from PTX ISA version 8.8" (§9.7.16.5.14) |
| `mma.sp` | same qualifiers | "requires sm_120a and are supported on sm_120f and later generation targets in the same family from PTX ISA version 8.8 except for `.kind::mxf4nvf4`/`.kind::mxf4`" (§9.7.16.6.3) |
| `mma.sp` | `.kind::mxf4nvf4`, `.kind::mxf4` | "are supported on following architectures: sm_120a sm_121a" (§9.7.16.6.3) |

Table 74 of the manual repeats this: `mma` and `mma.sp` rows "8.7 | sm_120a; 8.8 | sm_120f", and a separate `mma.sp` row "Kind `.kind::mxf4nvf4` and `.kind::mxf4` | 8.7 | sm_120a, sm_121a". [INFERENCE] Sparse FP4 block-scaled MMA is therefore architecture-specific (needs `sm_120a` on 12.0 and `sm_121a` on 12.1), whereas every dense kind is reachable through `sm_120f`.

CUTLASS strings (`cutlass/include/cute/arch/mma_sm120.hpp`): line 1809 `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.<A>.<B>.f32.ue8m0`; line 3159 `...kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64...ue8m0`; line 3216 `...scale_vec::4X.m16n8k64...ue4m3`; line 3137 `...scale_vec::4X...ue8m0`. Sparse (`mma_sm120_sparse.hpp:72, 2001, 3351`): `.sp::ordered_metadata` with `m16n8k64` (f8f6f4, mxf8f6f4) and `m16n8k128` (mxf4nvf4). CUTLASS does not emit `.kind::mxf4` anywhere in these two files.

### 1d. Compile guards

```
cutlass/include/cutlass/arch/config.h:156-172
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM120_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200)
    #define CUTLASS_ARCH_MMA_SM120_ENABLED 1
    #if (!defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM120_ALL))
      #define CUTLASS_ARCH_MMA_SM120A_ENABLED 1
    ...
    #if (!defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) && CUDA_ARCH_FAMILY(1200))
      #define CUTLASS_ARCH_MMA_SM120F_ENABLED CUTLASS_ARCH_MMA_SM120F_SUPPORTED
```

SM121 is the same pattern with `__CUDA_ARCH__ == 1210` and a toolkit floor of 12.9 (`config.h:228-246`). `CUTLASS_ARCH_MMA_SM120F_SUPPORTED` needs CUDA ≥ 12.9 (`config.h:166`). `CUDA_ARCH_FAMILY(X)` is `(__CUDA_ARCH_FAMILY_SPECIFIC__ == X)`, with the comment "`__CUDA_ARCH_FAMILY_SPECIFIC__` is introduced in CUDA 12.9" (`cutlass/include/cutlass/detail/helper_macros.hpp:222-226`).

```
cutlass/include/cute/arch/config.hpp:158-177
#if (defined(CUTLASS_ARCH_MMA_SM120_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM121_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED))
#  define CUTE_ARCH_MMA_SM120_ENABLED
#  define CUTE_ARCH_TMA_SM120_ENABLED
...
#    define CUTE_ARCH_F8F6F4_MMA_ENABLED
#    define CUTE_ARCH_MXF8F6F4_MMA_ENABLED
#    define CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED
#    define CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED
#  if (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)
#    define CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED
```

[INFERENCE] These CuTe macros key on the baseline `__CUDA_ARCH__` value, not on the `a`/`f` feature macro, so CuTe will emit the `.kind` strings even for a plain `-arch=sm_120` build; per the PTX target notes `ptxas` should then reject them. Not verified by compilation.

### 1e. wgmma and tcgen05 are not available

[PTX §9.7.17.5.2] `wgmma.mma_async` Target ISA Notes: "Requires sm_90a." Table 74 lists `wgmma.*` with a single supported target, `sm_90a`. [PTX §9.7.18.10.10.1] `tcgen05.mma`: "Supported on following architectures: sm_100a, sm_101a (Renamed to sm_110a...)" plus `sm_100f`/`sm_110f` families; `tcgen05.alloc` has the same list. No sm_12x target appears in either list.

CUTLASS mirrors this by equality, not ordering:

- `config.h:45`: `CUTLASS_ARCH_MMA_SM90_ENABLED` requires `__CUDA_ARCH__ == 900`; `CUTE_ARCH_MMA_SM90A_ENABLED` (the wgmma guard, `cutlass/include/cute/arch/mma_sm90_gmma.hpp:51`) derives only from `CUTLASS_ARCH_MMA_SM90A_ENABLED`.
- `config.h:89`: `CUTLASS_ARCH_MMA_SM100_ENABLED` requires `__CUDA_ARCH__ == 1000`.
- `cutlass/include/cute/arch/config.hpp:65-72`: `CUTE_ARCH_TCGEN05_*_MMA_ENABLED` and (`:140-143`) `CUTE_ARCH_TCGEN05_TMEM_ENABLED` are defined only for `SM100A/101A/103A/107A` (and the 100F/103F/107F/110 families); the SM120F/SM121F block (`:96-101`) defines only TMA and STSM macros.

Note `CUTLASS_ARCH_MMA_SM90_SUPPORTED` / `SM100_SUPPORTED` are toolkit-version tests only and are true in any sm_120 build; the `*_ENABLED` macros are the ones that carry the architecture test. [CUTLASS README:331-334] "kernels compiled for Blackwell SM100 architecture with arch conditional features (using `sm100a`) are not compatible with RTX 50 series GPUs."

---

## 2. TMA on SM120 / SM121

**Conclusion.** TMA (`cp.async.bulk.tensor`) and `mbarrier` are baseline features on sm_120/121, but without cluster multicast, and CUTLASS consumes TMA-loaded tiles through `ldmatrix` into registers.

[PTX §9.7.10.28.5.3 Target ISA Notes] "Requires sm_90 or higher. `.multicast::cluster` qualifier advised to be used with `.target` sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a." (no sm_12x target in the multicast list). `mbarrier.init`: "Requires sm_80 or higher." [PROG Table 32] "Tensor Memory Accelerator (TMA) unit: No (7.x, 8.x) | Yes (9.0, 10.x, 11.0, 12.x)"; Thread Block Cluster has the same row.

sm_120a restrictions [PTX §9.7.10.28.5.1]: "Following are the restrictions for sm_120a: `cp.async.bulk.tensor` with the direction `.shared::cluster.global` doesn't support: the sub-byte types; the qualifier `.swizzle_atomicity`".

CUTLASS specifics:

- TMA copies on SM120 are emitted with a CTA-local destination: `cutlass/include/cute/arch/copy_sm90_tma.hpp:59-61` `#if defined(CUTE_ARCH_TMA_SM120_ENABLED)` → `"cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"`, else the `.shared::cluster.global` form.
- `CUTE_ARCH_TMA_SM90_ENABLED` is defined for SM120 only under the `A` or `F` macros (`cutlass/include/cute/arch/config.hpp:56-63, 96-101`). [INFERENCE] CUTLASS TMA atoms therefore need `sm_120a`/`sm_120f`/`sm_121a` even though PTX accepts the instruction on baseline; this is a library policy.
- Mainloop collective `cutlass/include/cutlass/gemm/collective/sm120_mma_tma.hpp:118`: `using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;` and `:156` asserts `GmemTiledCopyA` is `SM90_TMA_LOAD` or `SM90_TMA_LOAD_MULTICAST`. `:153` asserts the MMA fragments are not `GMMA::DescriptorIterator`, i.e. operands are register fragments.
- Cluster: `sm120_mma_builder.inl:84` `static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");`. Example 79 header (`cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu:50-52`): "GeForce RTX 50 series GPUs do not support: 1. Multicast feature of TMA load. Cluster shape has to be 1x1x1. 2. Dynamic datatypes."
- Shared memory → registers: `sm120_mma_tma.hpp:511-523`:
  ```
  // copy smem->rmem for A/B operand
  copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
  copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));
  // Left shift A,B for FP4
  fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
  ...
  cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
  ```
  The copy atoms are `SM75_U32x4_LDSM_N` for 8-bit elements and `SM100_SU4_DU8x16_x4_LDSM_N` / `SM100_SU6_DU8x16_x4_LDSM_N` for FP4/FP6 (`cutlass/include/cutlass/gemm/collective/builders/sm120_common.inl:56-68`).
- Kernel layer: SM120 schedules derive from the Hopper ones (`cutlass/include/cutlass/gemm/dispatch_policy.hpp:604` `struct KernelTmaWarpSpecializedCooperativeSm120 : KernelTmaWarpSpecializedCooperative`) and so run `sm90_gemm_tma_warpspecialized_cooperative.hpp` / `..._pingpong.hpp`. Cooperative (`cutlass/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp:115-125`): "NumSchedThreads // 1 warp", "NumMMAThreads = size(TiledMma{}) // 8 warps", "NumMainloopLoadThreads // 1 warp", "NumEpilogueLoadThreads // 1 warp for C", `NumLoadWarpGroups = 1`, `MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup)` = 384. Producer warp roles: `Mainloop, Warp1, Epilogue, MainloopAux` (`:388-393`). The builder picks `AtomLayoutMNK` 4×2×1 (cooperative, 8 warps on one tile) or 2×2×1 (ping-pong, two 4-warp groups) at `sm120_mma_builder.inl:93-95`. Ping-pong asserts 128 MMA threads and 384 total (`sm90_gemm_tma_warpspecialized_pingpong.hpp:139-140`).
- One SM120-named kernel file exists, `cutlass/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`, with two mainloop load warps and separate A/B pipelines (`:131` "NumMainloopLoadThreads = NumThreadsPerWarp * 2; // 2 warp", `:119-124` "Asymmetric buffering ... AsymmetricKRatio = {1, 2} is the only support"). It is used by the sparse schedules (`dispatch_policy.hpp:1641`).
- Register reallocation: the cooperative kernel calls `warpgroup_reg_dealloc<LoadRegisterRequirement>()` / `warpgroup_reg_alloc<MmaRegisterRequirement>()` with 40/232 or 24/240 (`:150-151, 562, 772`). `setmaxnreg` is [PTX §9.7.21.5] "Supported on ... sm_90a, sm_100a, sm_101a, sm_120a ... sm_120f or higher in the same family"; CUTLASS activates it for `__CUDA_ARCH__ == 1200/1210` under the `ALL` or family macros (`cutlass/include/cutlass/arch/reg_reconfig.h:52-53, 62-63`).

---

## 3. `fence.proxy.async`

**Conclusion.** Any shared-memory location written by ordinary stores (generic proxy) and then read by TMA (async proxy) needs `fence.proxy.async` in the writing thread before the TMA instruction; the reverse direction is covered by an implicit fence at observed completion.

[PTX §8.6] "A memory proxy, or a proxy is an abstract label applied to a method of memory access. When two memory operations use distinct methods of memory access, they are said to be different proxies. ... A proxy fence is required to synchronize memory operations across different proxies." Non-generic methods include "async-proxy and tensormap-proxy accesses by `.async.bulk` operations".

[PTX §9.7.10.28.2] "The `cp{.reduce}.async.bulk` operations are performed in the asynchronous proxy (or async proxy). Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the async proxy, `fence.proxy.async` should be used to synchronize memory between generic proxy and the async proxy. The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit generic-async proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as soon as its completion is observed."

[PTX §9.7.15.4] "Bi-directional proxy fences ... take effect within a single thread and therefore do not have a `.scope` qualifier. Their cross-proxy synchronizing effect composes with other forms of synchronization". Target: "`fence.proxy.async` requires sm_90 or higher"; introduced in PTX ISA 8.0.

[INFERENCE] Consequences for an sm_120 attention kernel: (a) a TMA *load* followed by `ldmatrix` needs no explicit fence once the mbarrier wait has observed completion; (b) a TMA *store* of a tile written with `st`/`stmatrix` needs the fence in every writing thread, followed by a barrier so that the issuing thread is ordered after all fences; (c) because sm_120 `mma.sync` runs in the generic proxy, the Hopper-only `wgmma.fence` obligation disappears.

CUTLASS:

```
cutlass/include/cutlass/arch/barrier.h:787-795
void fence_view_async_shared() { ... "fence.proxy.async.shared::cta; \n" ...
cutlass/include/cutlass/arch/barrier.h:768-776
// Note : It must be composed with an appropriate sync instruction with the right scope
// to ensure visibility eg. __syncthreads() or a cluster_arrive() + cluster_wait()
void fence_barrier_init() { ... "fence.mbarrier_init.release.cluster; \n" ...
```

Both are compiled only when `CUDA_BARRIER_ENABLED`, defined as `__CUDA_ARCH__ >= 900 && (__CUDACC_VER_MAJOR__ >= 12)` (`barrier.h:43`), so they are live on baseline sm_120.

Call sites on the SM120 path (the SM120 epilogue builder instantiates the SM90 epilogue: `cutlass/include/cutlass/epilogue/collective/builders/sm120_builder.inl:422`):

- `cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp:736-737`: `cutlass::arch::fence_view_async_shared(); // ensure smem writes are visible to TMA` then `synchronize(); // ensure all threads have issued their async fence`.
- same file `:817`: fence before `load_pipeline.consumer_release(...)`, with comment "Let producer load warp know smem buffers are consumed and empty".
- `cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:341`: `fence_barrier_init()` at the end of `PipelineTmaAsync::init_barriers`.
- `cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp:479`: fence after a generic store of the work-tile record and before `producer_commit`.
- `sm120_mma_tma.hpp` itself contains no fence call (load-only TMA).

---

## 4. Shared memory and other limits

[TUNE §1.4.1.1], verbatim:

- "The maximum number of concurrent warps per SM is 64 for compute capability 10.0 and 48 for compute capability 12.0."
- "The register file size is 64K 32-bit registers per SM."
- "The maximum number of registers per thread is 255."
- "The maximum number of thread blocks per SM is 32 for devices of compute capability 10.0 and 12.0."
- "For devices of compute capability 10.0 shared memory capacity per SM is 228 KB. For devices of compute capability 12.0, shared memory capacity per SM is 128KB."
- "For devices of compute capability 10.0 the maximum shared memory per thread block is 227 KB. For devices of compute capability 12.0 the maximum shared memory per thread block is 99 KB."

[TUNE §1.4.1.2] "The maximum portable cluster size supported is 8; however, NVIDIA Blackwell B200 GPU allows for a nonportable cluster size of 16 by opting in." [TUNE §1.4.2.3] "static shared memory allocations remain limited to 48 KB, and an explicit opt-in is also required to enable dynamic allocations above this limit."

The tuning guide says nothing about compute capability 12.1 (its revision history reads "Added support for compute capability 10.0 and compute capability 12.0"), nothing about TMA, and nothing distinguishing tensor cores on consumer versus datacenter parts: **not found**.

[PROG Tables 33-35] use a single "12.x" column: max threads per block 1024; resident blocks per SM 24; resident warps per SM 48; resident threads per SM 1536; registers per thread 255; max shared memory per SM 100 KB; per thread block 99 KB; "12.x | Unified Data Cache Size 128 KB | SMEM Capacity Sizes 0, 8, 16, 32, 64, 100". Table 31 puts 12.1 in the `compute_120f` family.

Two documented disagreements, unresolved:

1. Shared memory per SM for 12.0: TUNE says 128 KB, PROG says 100 KB (128 KB is PROG's unified data cache size). The per-block figure, 99 KB, agrees in both.
2. Thread blocks per SM for 12.x: TUNE says 32, PROG says 24.

[CUTLASS] `cutlass/include/cutlass/arch/arch.h:46-48`: `sm100_smem_capacity_bytes = 232448; ... sm120_smem_capacity_bytes = 101376;` (101376 B = 99 KiB), used for both SM120 and SM121 since `arch::Sm120` is the only tag (`arch.h:118-121`). [INFERENCE] With 384 threads and 99 KiB, one CTA per SM is the design point (`MinBlocksPerMultiprocessor = 1`).

---

## 5. Compilation targets

[PTX §11.1.2] "Target architectures with suffix "a", such as sm_90a, include architecture-specific features that are supported on the specified architecture only ... PTX code generated for such targets cannot be run on later generation devices." "Target architectures with suffix "f", such as sm_100f, include family-specific features that are supported only within the same architecture family. ... Family-specific features can be used with f-targets as well as a-targets of later generation devices in the same family." Table 70: "sm_12x family | sm_120f, sm_121f, future targets in sm_12x family".

[PROG Table 31] "compute_120f | 12.0 | 12.1" and "compute_121f | 12.1".

Introduction [PTX §11.1.2 notes and release table]: "Target string sm_120 introduced in PTX ISA version 8.7. ... sm_120f ... 8.8. ... sm_120a ... 8.7. ... sm_121 ... 8.8. ... sm_121f ... 8.8. ... sm_121a ... 8.8." PTX ISA 8.7 = "CUDA 12.8, driver r570"; PTX ISA 8.8 = "CUDA 12.9, driver r575".

| Target | First toolkit | Runs on | Adds |
|---|---|---|---|
| `sm_120` | CUDA 12.8 | 12.0, 12.1 [INFERENCE from onion model] | baseline: sm_80 `mma` (bf16/f16), sm_89 FP8 `mma`, TMA, mbarrier, clusters, `fence.proxy.async`, `stmatrix` m8n8 |
| `sm_120a` | CUDA 12.8 | 12.0 only | `.kind::*` mma, FP6/FP4 types, `mma.sp` mxf4/mxf4nvf4, `ldmatrix` m16n16/m8n16/.b8, `stmatrix` m16n8, `setmaxnreg`, `tensormap.replace` |
| `sm_120f` | CUDA 12.9 | 12.0, 12.1 | the same list except `mma.sp` `.kind::mxf4nvf4`/`.kind::mxf4` |
| `sm_121` | CUDA 12.9 | 12.1 | baseline |
| `sm_121f` | CUDA 12.9 | 12.1 | family features |
| `sm_121a` | CUDA 12.9 | 12.1 only | adds `mma.sp` mxf4/mxf4nvf4 |

The PTX manual does not enumerate any feature that is on `sm_121a` and absent from `sm_120a`: **not found**. Open inconsistency: several PTX lists (`setmaxnreg`, `ldmatrix` shapes, `tensormap.replace`) name `sm_120a` but not `sm_121a`; `sm_121a` reaches them only through the clause "family-specific features can be used with ... a-targets of later generation devices in the same family".

[CUTLASS] `cutlass/CMakeLists.txt:189` `list(APPEND CUTLASS_NVCC_ARCHS_SUPPORTED 100 100a 120 120a)` and `:198` `... 121 121a 100f 120f 121f 103a 103f`. Example 79 builds for `"120a|120f|121a"` (`cutlass/examples/79_blackwell_geforce_gemm/CMakeLists.txt:49`). README table (`cutlass/README.md:299-300`): "NVIDIA GeForce RTX 50x0 series | 12.0 | 12.8" and "NVIDIA DGX Spark | 12.1 | 13.0" (minimum toolkit); this 13.0 figure is stricter than `config.h:228`, which enables SM121 from 12.9.

---

## 6. Hardware facts

| Item | Value | Source |
|---|---|---|
| GB10 (DGX Spark) compute capability | 12.1 | https://developer.nvidia.com/cuda/gpus row "NVIDIA GB10 (DGX Spark)"; `cutlass/README.md:300` |
| GB10 CUDA cores | "CUDA Cores: 6,144" | https://docs.nvidia.com/dgx/dgx-spark/hardware.html |
| GB10 SM count | **not found** in a primary source. 6,144 / 128 = 48 is [INFERENCE] assuming 128 CUDA cores per SM | — |
| GB10 memory | "128 GB LPDDR5x unified system memory, 256-bit interface, 4266 MHz, 273 GB/s bandwidth" | https://docs.nvidia.com/dgx/dgx-spark/hardware.html |
| GB10 GPU description | "NVIDIA Blackwell Architecture with 5th Generation Tensor Cores, 4th Generation RT Cores" | same page |
| RTX PRO 6000 Blackwell compute capability | 12.0 (Workstation, Max-Q and Server editions) | https://developer.nvidia.com/cuda/gpus |
| RTX PRO 6000 Blackwell memory | "96 GB GDDR7 with error-correcting code (ECC)" | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ |
| RTX PRO 6000 Blackwell bandwidth | "1792 GB/sec" | same page |
| RTX PRO 6000 Blackwell CUDA cores | 24,064. Not on the NVIDIA product page. **Third-party hosting**: NVIDIA-branded datasheet on pny.com, seen only as a search-result snippet, not opened | https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/NVIDIA%20RTX/English/NVIDIA-RTX-PRO-6000-Blackwell-Workstation-Edition.pdf |
| RTX PRO 6000 SM count | **not found**. 24,064 / 128 = 188 is [INFERENCE] | — |
| GeForce RTX 5090 compute capability | 12.0 | https://developer.nvidia.com/cuda/gpus |

The product pages were read through a summarizing fetch tool, so the quoted strings are the tool's extraction rather than a byte-exact read.

---

## 7. `ldmatrix` and register pressure

[PTX §9.7.16.5.15 syntax]

```
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];
ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];
ldmatrix.sync.aligned.m8n16.num{.ss}.dtype.ctype            r, [p];
.shape = {.m8n8, .m16n16}; .num = {.x1, .x2, .x4}; .type = {.b16, .b8};
.dst_fmt = { .b8x16 }; .src_fmt = { .b6x16_p32, .b4x16_p64 }; .dtype = { .s8 }; .ctype = { .s4 };
```

Shape table: `.m8n8` 8x8 16-bit; `.m16n16` 16x16 "8-bit or 6-bit or 4-bit"; `.m8n16` 8x16 "6-bit or 4-bit". "for 16x16 matrices, `.trans` is mandatory."

[PTX Target ISA Notes] "Requires sm_75 or higher. Shapes `.m16n16`, `.m8n16` are supported on following architectures: sm_100a, sm_101a (Renamed to sm_110a...), sm_120a. And are supported on following family-specific architectures from PTX ISA version 8.8: sm_100f ..., sm_120f or higher in the same family, sm_110f ...". The same lists apply to "Type `.b8` with ldmatrix" and "Qualifiers `.src_fmt`, `.dst_fmt`". "Types `.s8.s4` are supported on ... sm_90a, sm_100f ..., sm_110f ..., sm_120f or higher in the same family" (PTX ISA 9.4). `stmatrix`: "Requires sm_90 or higher. Shape `.m16n8` is supported on ... sm_120a ... sm_120f or higher in the same family".

[INFERENCE] For FP8 (8-bit) operands in K-major (row) layout, the classic `ldmatrix.m8n8.x4.b16` is the load path: four FP8 values travel as two 16-bit elements. CUTLASS does exactly this, selecting `SM75_U32x4_LDSM_N` for 8-bit elements (`sm120_common.inl:64`). The new `.m16n16.trans.b8` form (`SM100_U8x8_LDSM_T`, `cutlass/include/cute/arch/copy_sm100.hpp:107`) is the transposing 8-bit load. FP4/FP6 use `ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` / `.b6x16_p32` (`copy_sm100.hpp:259, 281`), guarded by `CUTE_ARCH_LDSM_SM100A_ENABLED`, which `cute/arch/config.hpp:132-138, 195-199` defines for SM120A/SM121A/SM120F/SM121F.

FP4 quirk [CUTLASS `cutlass/include/cute/atom/mma_traits_sm120.hpp:211-217`]: "ld.matrix b4x16_p64 places FP4 data at the first four bits in each eight-bit container, whereas MMA F8F6F4 expects the four-bit data to be in the middle of the eight-bit container. Thus, e2m1 operands being fed to MMA F8F6F4 must be shifted left by two bits. ... NOTE: Same transformation is NOT needed for FP6 and FP8." This matches [PTX mma description] "The matrix element of type `.e2m1` resides in central 4 bits of the 8-bit container".

Register limits: 255 per thread [TUNE; PROG Table 34]; `setmaxnreg` operand "must be in the range 24 to 256 (both inclusive) and must be a multiple of 8" [PTX §9.7.21.5]. CUTLASS estimates accumulator pressure as `RegsPerThread = size<0>(TileShape) * size<1>(TileShape) / NumMMAThreads * sizeof(ElementAccumulator) / sizeof(uint32_t)` and treats `>= 208` (or `>= 128` for SM120 block-scaled) as heavy, with the comment "Detect if this is SM120 blockscaled kernel which hits high register pressure on smaller tiles" (`sm90_gemm_tma_warpspecialized_cooperative.hpp:132-151`). [INFERENCE] Unlike Hopper, where A and B stay in shared memory behind descriptors, every sm_120 `mma.sync` also holds 4 A registers and 2 B registers per atom per thread, so operand fragments compete with accumulators for the 255-register budget.

---

## Quick reference: capability × architecture

| Capability | SM90 (Hopper) | SM100 (datacenter Blackwell) | SM120 / SM121 |
|---|---|---|---|
| Warp-level `mma.sync` bf16/f16 m16n8k16 | Yes, "requires sm_80 or higher" [PTX §9.7.16.5.14] | Yes, same note | Yes, same note; `mma_sm80.hpp:39` |
| `mma.sync` FP8, no `.kind` | Yes, "requires sm_89 or higher" [PTX] | Yes | Yes; `mma_sm89.hpp:50,79` |
| `mma.sync` `.kind::f8f6f4`, FP6/FP4 | No: "requires sm_120a ... sm_120f" [PTX] | No (same note) | Yes with `sm_120a`/`sm_120f`; `mma_sm120.hpp:70` |
| `mma.sync` block-scaled (mxf8f6f4, mxf4, mxf4nvf4) | No | No (uses `tcgen05.mma` kinds) | Dense: `sm_120a`/`sm_120f`. Sparse mxf4/mxf4nvf4: `sm_120a`, `sm_121a` only [PTX §9.7.16.6.3] |
| `wgmma.mma_async` | "Requires sm_90a" [PTX §9.7.17.5.2] | No (Table 74: sm_90a only) | No; `config.h:45`, `mma_sm90_gmma.hpp:51` |
| `tcgen05.*` / tensor memory | No | "sm_100a ... sm_100f or higher" [PTX §9.7.18.10.10.1] | No; `cute/arch/config.hpp:65-72, 96-101` |
| TMA `cp.async.bulk.tensor` | "Requires sm_90 or higher" [PTX] | Yes | Yes; `sm120_mma_tma.hpp:156`; emitted as `.shared::cta.global` (`copy_sm90_tma.hpp:59`) |
| TMA multicast | "advised ... sm_90a" [PTX] | sm_100a / sm_100f [PTX] | Not listed [PTX]; "no programmatic multicast on this arch" `sm120_mma_builder.inl:84` |
| `mbarrier` | "Requires sm_80 or higher" [PTX §9.7.15.16.12] | Yes | Yes; `PipelineTmaAsync`, `sm120_mma_tma.hpp:118` |
| `fence.proxy.async` | "requires sm_90 or higher" [PTX §9.7.15.4] | Yes | Yes; `barrier.h:43, 787` |
| `setmaxnreg` | sm_90a [PTX §9.7.21.5] | sm_100a / sm_100f | sm_120a / sm_120f; `reg_reconfig.h:52-53` |
| `ldmatrix` m8n8 b16 | "Requires sm_75 or higher" [PTX] | Yes | Yes; `sm120_common.inl:64` |
| `ldmatrix` m16n16 / m8n16 / `.b8` / sub-byte formats | No | sm_100a / sm_100f [PTX] | sm_120a / sm_120f [PTX]; `copy_sm100.hpp:171-281` |
| Operand path to MMA | smem descriptor | smem descriptor or TMEM | smem → `ldmatrix` → registers; `sm120_mma_tma.hpp:511-523` |
| Max smem per thread block | 227 KB [PROG Table 34] | 227 KB [TUNE §1.4.1.1] | 99 KB [TUNE §1.4.1.1; PROG Table 34]; `arch.h:48` = 101376 B |
| Smem per SM | 228 KB [PROG] | 228 KB [TUNE] | 128 KB [TUNE] vs 100 KB [PROG] (conflict) |
| Max warps per SM | 64 [PROG Table 33] | 64 [TUNE] | 48 [TUNE; PROG] |
| Registers per thread | 255 [PROG Table 34] | 255 [TUNE] | 255 [TUNE; PROG] |
| Cluster size | portable 8 [TUNE §1.4.1.2] | portable 8, B200 non-portable 16 [TUNE] | Clusters supported [PROG Table 32]; CUTLASS requires 1×1×1 (`sm120_mma_builder.inl:84`); hardware maximum **not found** |
| Family target | none (`sm_90a` only) | `sm_100f` → 10.0, 10.3, 10.7 [PROG Table 31] | `sm_120f` → 12.0, 12.1; `sm_121f` → 12.1 [PROG Table 31] |

---

## Unresolved, and the smallest experiments that would settle each

1. **Does baseline `sm_120` accept `.kind::f8f6f4`?** Expected no. Test: assemble a one-instruction PTX module containing `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` three times, with `.target sm_120`, `sm_120f`, `sm_120a`; only the first should fail in `ptxas`.
2. **Does `sm_120f` code using dense `.kind::mxf4nvf4` load on a GB10?** PTX says yes for dense, no for sparse. Test: build CUTLASS example 79b with `-DCUTLASS_NVCC_ARCHS=120f` and run on DGX Spark; then repeat with a `sm120_bssp_gemm_f4_f4_f32` sparse test, which should fail to assemble.
3. **Per-SM shared memory on 12.0 and 12.1 (100 KB vs 128 KB) and blocks per SM (24 vs 32).** Test: `cudaDeviceGetAttribute` for `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, `cudaDevAttrMaxSharedMemoryPerBlockOptin`, and `cudaDevAttrMaxBlocksPerMultiprocessor` on an RTX PRO 6000 and on a GB10.
4. **SM counts.** Test: `cudaDevAttrMultiProcessorCount` on each device (expected 188 and 48 if the 128-cores-per-SM inference holds).
5. **Whether 12.1 limits differ from 12.0 at all.** No primary source separates them; the queries in items 3-4 on a GB10 are the discriminating check.

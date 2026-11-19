"""Selected-position specialization of the CuTe paged-forward engine.

The engine supports direct BF16 output for large QSA prefills and FP32 split
partials for latency-sensitive small launches. Cache strides specialize the
compiled callable so both entries consume zero-copy per-layer views from an
interleaved BLHNC allocation without carrying layout arithmetic as runtime
state. Every page-scaled cache offset is widened to Int64 before multiplication.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from threading import RLock

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
import torch
from cutlass import (
    BFloat16,
    Float32,
    Float8E4M3FN,
    Int32,
    Int64,
    Uint32,
    const_expr,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import warp, warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from b12x.attention._shared.contiguous import layout_utils
from b12x.attention._shared.contiguous.forward import (
    warp_mma_gemm,
)
from b12x.attention._shared.contiguous.softmax import Softmax
from b12x.attention._shared.cute.ops import fmax
from b12x.attention._shared.cute import ops as _cute_ops
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compiler import run_compiled
from b12x._lib.compile_plan import compile_only_launches_enabled, program_keys, record_program
from b12x._lib.intrinsics import (
    cp_async_bulk_g2s_mbar,
    fp8x4_e4m3_to_bfloat2x2_native_sm120,
    ld_global_nc_v4_u32,
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._selected_forward_config import HEAD_DIM as _HEAD_DIM
from ._selected_forward_config import MAX_SPLITS as _NUM_SPLITS
from ._selected_forward_config import SELECTION_WIDTH as _SELECTION_WIDTH


_THREADS = 128
_WARPS = _THREADS // 32
# Experimental software pipeline for the FP8 shared-stride path (see _pipelined_step).
_PIPELINE = os.environ.get("B12X_SELECTED_PIPELINE", "0") == "1"
# Experimental TMA (cp.async.bulk + mbarrier) gather for the FP8 shared-stride path (see _tma_step).
_TMA = os.environ.get("B12X_SELECTED_TMA", "0") == "1"
# TMA mode for V: 1 = bulk-copy V too (4 x 64 B per row, in place); 0 = register prefetch for V.
_TMA_V = os.environ.get("B12X_SELECTED_TMA_V", "0") == "1"
# Split QK + online softmax across all warps (pipelined mode only): each warp owns tile_n/4 columns.
_SPLITQK = os.environ.get("B12X_SELECTED_SPLITQK", "0") == "1"
# Experimental: Q/K/V smem atom swizzle S<3,3,3> (16-byte units for BF16 element offsets).
_SWZ333 = os.environ.get("B12X_SELECTED_SWZ333", "0") == "1"
# Experimental: gather thread mapping with rows varying fastest (conflict-free swizzled stores).
_ROWMAP = os.environ.get("B12X_SELECTED_ROWMAP", "0") == "1"
_RING = 3  # position-base ring: tile i (mask), i+1 (gather in flight), i+2 (being resolved)
_LOG2_E = 1.4426950408889634
_LOCK = RLock()
_KERNEL_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}
_MERGE_CACHE: dict[tuple[object, ...], Callable[..., None]] = {}
_MERGE_WARMED: dict[tuple[object, ...], Callable[..., None]] = {}


def _add(left: Float32, right: Float32) -> Float32:
    return left + right


@dsl_user_op
def _exp2_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pointer_as_int64(pointer, offset, *, loc=None, ip=None) -> Int64:
    element_pointer = pointer + offset
    return Int64(llvm.ptrtoint(T.i64(), element_pointer.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def _cp_async_load_128b_zfill(
    smem_addr: Int32,
    gmem_addr: Int64,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
            Int32(src_bytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _SelectedPositionPagedForwardKernel:
    """Paged-forward entry over caller-selected logical K/V positions."""

    def __init__(
        self,
        *,
        q_heads: int,
        kv_heads: int,
        kv_is_fp8: bool,
        direct_output: bool,
        return_lse: bool,
        kv_warps: int,
        page_size: int,
        key_strides: tuple[int, int, int],
        value_strides: tuple[int, int, int],
        selection_width: int = _SELECTION_WIDTH,
    ) -> None:
        self.q_heads = int(q_heads)
        self.selection_width = int(selection_width)
        if self.selection_width < _SELECTION_WIDTH:
            raise ValueError("selected-position capacity must cover the QSA selection")
        self.kv_heads = int(kv_heads)
        self.heads_per_kv = self.q_heads // self.kv_heads
        self.kv_is_fp8 = bool(kv_is_fp8)
        self.direct_output = bool(direct_output)
        self.return_lse = bool(return_lse)
        self.kv_warps = int(kv_warps)
        if self.kv_warps not in (1, 2, 4):
            raise ValueError(
                "selected-position paged forward requires 1, 2, or 4 KV warps"
            )
        self.tile_n = 16 * self.kv_warps
        self.page_size = int(page_size)
        if self.page_size <= 0:
            raise ValueError(
                "selected-position paged forward page_size must be positive"
            )
        self.key_page_stride, self.key_token_stride, self.key_head_stride = map(
            int, key_strides
        )
        (
            self.value_page_stride,
            self.value_token_stride,
            self.value_head_stride,
        ) = map(int, value_strides)
        self.shared_cache_strides = tuple(map(int, key_strides)) == tuple(
            map(int, value_strides)
        )
        self.qk_warps = 1 if self.heads_per_kv <= 16 else 2
        self.pv_warps = self.qk_warps * 4
        self.tile_m = self.qk_warps * 16
        self.threads = self.pv_warps * 32
        # Threads outside the QK warps resolve the next-next tile's positions while QK runs.
        self.resolver_offset = self.qk_warps * 32
        self.pipelined = bool(
            _PIPELINE
            and self.kv_is_fp8
            and self.shared_cache_strides
            and self.resolver_offset + self.tile_n <= self.threads
            and self.threads == 128  # gather mapping: thread // 8 indexes the 16 vectors of a row
        )
        self.gather_iters = self.tile_n * (_HEAD_DIM // 16) // self.threads
        # TMA mode: K(i+1) is bulk-copied as FP8 into the Q tile's smem (free after the first QK,
        # Q then lives in registers); V(i+1) is bulk-copied as FP8 into the upper 64 B of each
        # 128-byte segment of the BF16 V tile and expanded in place. Needs one copy lane per row.
        self.tma = bool(
            _TMA
            and self.kv_is_fp8
            and self.shared_cache_strides
            and self.qk_warps == 1
            and self.tile_n <= 32
            and self.tile_n * _HEAD_DIM <= self.tile_m * _HEAD_DIM * 2
        )
        self.segments = self.tile_n * (_HEAD_DIM // 64)  # 64-column (128-byte BF16) segments per tile
        self.tma_value = bool(_TMA_V)
        if self.tma:
            self.pipelined = False
        self.split_qk = bool(
            _SPLITQK and self.pipelined and self.qk_warps == 1 and self.tile_n % 32 == 0
        )
        # Threads that hold QK fragments: warp 0 normally, every warp in split mode.
        self.qk_threads = self.threads if self.split_qk else self.qk_warps * 32

    def _tiled_mma_qk(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (1, 4, 1) if self.split_qk else (self.qk_warps, 1, 1),
            permutation_mnk=(self.tile_m, self.tile_n, 16),
        )

    def _tiled_mma_pv(self) -> cute.TiledMma:
        return cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)),
            (self.qk_warps, 4, 1),
            permutation_mnk=(self.tile_m, _HEAD_DIM, 16),
        )

    @cute.jit
    def _load_fp8_vector_to_bf16_shared(
        self,
        source: cute.Pointer,
        source_base: Int64,
        destination: cute.Tensor,
        token: Int32,
        dimension_base: Int32,
    ) -> None:
        x0 = Uint32(0)
        x1 = Uint32(0)
        x2 = Uint32(0)
        x3 = Uint32(0)
        if source_base >= Int64(0):
            x0, x1, x2, x3 = ld_global_nc_v4_u32(
                _pointer_as_int64(source, source_base + dimension_base.to(Int64))
            )
        y00, y01 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x0)
        y10, y11 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x1)
        y20, y21 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x2)
        y30, y31 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x3)
        destination0 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx((token, dimension_base), destination.layout)
        )
        destination1 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx(
                (token, dimension_base + Int32(8)),
                destination.layout,
            )
        )
        st_shared_v4_u32(destination0, y00, y01, y10, y11)
        st_shared_v4_u32(destination1, y20, y21, y30, y31)

    # ---- software pipeline (FP8, shared K/V strides) -------------------------------------
    @cute.jit
    def _resolve_tile_positions(
        self,
        tile: Int32,
        slot: Int32,
        column: Int32,
        row: Int32,
        kv_head: Int32,
        request_id: Int64,
        request_valid,
        query_position: Int64,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        base_ring: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
    ) -> None:
        """Write tile's K/V row bases into ring slot; -1 marks invalid/padded columns."""
        selected_column = tile * Int32(self.tile_n) + column
        valid = request_valid & (selected_column < Int32(self.selection_width))
        logical_position = Int64(-1)
        if valid:
            selected_offset = row.to(Int64) * Int64(self.selection_width) + selected_column.to(Int64)
            logical_position = selected_positions[selected_offset].to(Int64)
            valid = (logical_position >= Int64(0)) & (logical_position <= query_position)
        physical_page = Int64(-1)
        page_offset = Int64(0)
        if valid:
            logical_page = logical_position // Int64(self.page_size)
            valid = logical_page < table_width
            if valid:
                table_offset = request_id * table_width + logical_page
                physical_page = block_table[table_offset].to(Int64)
                valid = (physical_page >= Int64(0)) & (physical_page < num_cache_pages)
                page_offset = logical_position % Int64(self.page_size)
        key_base = Int64(-1)
        if valid:
            key_base = (
                physical_page * Int64(self.key_page_stride)
                + page_offset * Int64(self.key_token_stride)
                + kv_head.to(Int64) * Int64(self.key_head_stride)
            )
        base_ring[slot * Int32(self.tile_n) + column] = key_base

    @cute.jit
    def _issue_fp8_gather(
        self,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        base_ring: cute.Tensor,
        slot: Int32,
        thread: Int32,
        staged: cute.Tensor,
    ) -> None:
        """Start the global loads for one tile into registers (no shared-memory traffic)."""
        vectors_per_row = _HEAD_DIM // 16
        for vector_iter in cutlass.range_constexpr(self.gather_iters):
            if const_expr(_ROWMAP):
                # Rows vary fastest across each 8-lane phase: with 16-byte-unit swizzle the chunk
                # is XORed with (row % 8), so 8 rows at one column hit 8 different bank groups.
                token = thread % Int32(8) + Int32(8 * vector_iter)
                dimension_base = (thread // Int32(8)) * Int32(16)
            else:
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
            base = Int64(base_ring[slot * Int32(self.tile_n) + token])
            k0 = Uint32(0)
            k1 = Uint32(0)
            k2 = Uint32(0)
            k3 = Uint32(0)
            v0 = Uint32(0)
            v1 = Uint32(0)
            v2 = Uint32(0)
            v3 = Uint32(0)
            if base >= Int64(0):
                k0, k1, k2, k3 = ld_global_nc_v4_u32(
                    _pointer_as_int64(key_cache, base + dimension_base.to(Int64))
                )
                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                    _pointer_as_int64(value_cache, base + dimension_base.to(Int64))
                )
            staged[vector_iter * 8 + 0] = k0
            staged[vector_iter * 8 + 1] = k1
            staged[vector_iter * 8 + 2] = k2
            staged[vector_iter * 8 + 3] = k3
            staged[vector_iter * 8 + 4] = v0
            staged[vector_iter * 8 + 5] = v1
            staged[vector_iter * 8 + 6] = v2
            staged[vector_iter * 8 + 7] = v3

    @cute.jit
    def _store_fp8_one(
        self, destination: cute.Tensor, token: Int32, dimension_base: Int32, x0, x1, x2, x3
    ) -> None:
        y00, y01 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x0)
        y10, y11 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x1)
        y20, y21 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x2)
        y30, y31 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x3)
        destination0 = shared_ptr_to_u32(
            destination.iterator + cute.crd2idx((token, dimension_base), destination.layout)
        )
        destination1 = shared_ptr_to_u32(
            destination.iterator
            + cute.crd2idx((token, dimension_base + Int32(8)), destination.layout)
        )
        if const_expr(_SWZ333):
            # Lanes of one 8-lane phase cover two 64-column blocks. Storing the odd 8-column chunk
            # first in odd blocks makes the first (and second) stores of the phase hit disjoint
            # bank-group parities after the (row % 8) XOR, removing the 2-way conflict.
            odd = ((dimension_base // Int32(64)) % Int32(2)) == Int32(1)
            first = cutlass.select_(odd, destination1, destination0)
            second = cutlass.select_(odd, destination0, destination1)
            st_shared_v4_u32(
                first,
                cutlass.select_(odd, y20, y00), cutlass.select_(odd, y21, y01),
                cutlass.select_(odd, y30, y10), cutlass.select_(odd, y31, y11),
            )
            st_shared_v4_u32(
                second,
                cutlass.select_(odd, y00, y20), cutlass.select_(odd, y01, y21),
                cutlass.select_(odd, y10, y30), cutlass.select_(odd, y11, y31),
            )
        else:
            st_shared_v4_u32(destination0, y00, y01, y10, y11)
            st_shared_v4_u32(destination1, y20, y21, y30, y31)

    @cute.jit
    def _store_fp8_gather(
        self,
        staged: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        thread: Int32,
    ) -> None:
        """Convert staged FP8 registers to BF16 and write the K/V tiles (same layout as before)."""
        vectors_per_row = _HEAD_DIM // 16
        for vector_iter in cutlass.range_constexpr(self.gather_iters):
            if const_expr(_ROWMAP):
                # Rows vary fastest across each 8-lane phase: with 16-byte-unit swizzle the chunk
                # is XORed with (row % 8), so 8 rows at one column hit 8 different bank groups.
                token = thread % Int32(8) + Int32(8 * vector_iter)
                dimension_base = (thread // Int32(8)) * Int32(16)
            else:
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
            self._store_fp8_one(
                shared_key, token, dimension_base,
                staged[vector_iter * 8 + 0], staged[vector_iter * 8 + 1],
                staged[vector_iter * 8 + 2], staged[vector_iter * 8 + 3],
            )
            self._store_fp8_one(
                shared_value, token, dimension_base,
                staged[vector_iter * 8 + 4], staged[vector_iter * 8 + 5],
                staged[vector_iter * 8 + 6], staged[vector_iter * 8 + 7],
            )

    @cute.jit
    def _pipelined_step(
        self,
        local_tile: Int32,
        split: Int32,
        splits: Int32,
        row: Int32,
        kv_head: Int32,
        request_id: Int64,
        request_valid,
        query_position: Int64,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        base_ring: cute.Tensor,
        staged: cute.Tensor,
        reduce_max: cute.Tensor,
        position_ring: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        shared_probability: cute.Tensor,
        shared_row_scale: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
        thread: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        register_query: cute.Tensor,
        register_key: cute.Tensor,
        register_probability: cute.Tensor,
        register_value: cute.Tensor,
        copy_query: cute.TiledCopy,
        copy_key: cute.TiledCopy,
        copy_probability: cute.TiledCopy,
        copy_value: cute.TiledCopy,
        copy_source_query: cute.Tensor,
        copy_source_key: cute.Tensor,
        copy_source_probability: cute.Tensor,
        copy_source_value: cute.Tensor,
        accumulator_output: cute.Tensor,
        softmax: Softmax,
        is_first: cutlass.Constexpr,
    ) -> None:
        """One pipelined tile. Entry invariant: K/V(i) in smem, ring slots i and i+1 resolved.

        1. all threads issue global loads for tile i+1 into registers
        2. QK warps: scores, mask (ring slot i), online softmax, P -> smem
           resolver threads: positions of tile i+2 -> ring slot i+2
        3. sync; all warps: P @ V(i)
        4. sync; all threads: convert + store tile i+1 into the K/V tiles; sync
        """
        slot = local_tile % Int32(_RING)
        next_slot = (local_tile + Int32(1)) % Int32(_RING)
        next_next_slot = (local_tile + Int32(2)) % Int32(_RING)

        self._issue_fp8_gather(key_cache, value_cache, base_ring, next_slot, thread, staged)
        # Split mode: deferred two-stage position resolution, latency hidden behind QK and PV.
        #   now:  load selected position for tile i+3; look up the block table for tile i+2
        #         (its position was loaded during tile i-1 and parked in position_ring)
        #   end:  write tile i+2's row bases and park tile i+3's position
        resolve_column = thread - Int32(self.resolver_offset)
        is_resolver = (resolve_column >= Int32(0)) & (resolve_column < Int32(self.tile_n))
        next_position = Int32(-1)
        resolved_page = Int64(-1)
        resolved_position = Int64(0)
        if const_expr(self.split_qk):
            if is_resolver:
                selected_column = (split + (local_tile + Int32(3)) * splits) * Int32(self.tile_n) + resolve_column
                if request_valid & (selected_column < Int32(self.selection_width)):
                    next_position = selected_positions[
                        row.to(Int64) * Int64(self.selection_width) + selected_column.to(Int64)
                    ]
                position = Int64(position_ring[next_next_slot * Int32(self.tile_n) + resolve_column])
                if request_valid & (position >= Int64(0)) & (position <= query_position):
                    logical_page = position // Int64(self.page_size)
                    if logical_page < table_width:
                        resolved_page = block_table[request_id * table_width + logical_page].to(Int64)
                        resolved_position = position

        if thread < Int32(self.qk_threads):
            accumulator_score = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C((self.tile_m, self.tile_n)),
                Float32,
            )
            accumulator_score.fill(0.0)
            warp_mma_gemm(
                tiled_mma_qk,
                accumulator_score,
                register_query,
                register_key,
                copy_source_query,
                copy_source_key,
                copy_query,
                copy_key,
                A_in_regs=not is_first,
            )
            score_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    if (coordinate[0] >= Int32(self.heads_per_kv)) | (
                        Int64(base_ring[slot * Int32(self.tile_n) + coordinate[1]]) < Int64(0)
                    ):
                        score_mn[m, n] = -Float32.inf
            if const_expr(self.split_qk):
                row_scale = self._split_online_softmax(
                    accumulator_score, softmax, score_coordinates, reduce_max, thread, is_first
                )
            else:
                row_scale = softmax.online_softmax(accumulator_score, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    shared_probability[coordinate[0], coordinate[1]] = BFloat16(score_mn[m, n])
                if score_coordinates[m, 0][1] == Int32(0):
                    shared_row_scale[score_coordinates[m, 0][0]] = row_scale[m]
        else:
            column = thread - Int32(self.resolver_offset)
            if column < Int32(self.tile_n):
                self._resolve_tile_positions(
                    split + (local_tile + Int32(2)) * splits,
                    next_next_slot,
                    column,
                    row,
                    kv_head,
                    request_id,
                    request_valid,
                    query_position,
                    block_table,
                    selected_positions,
                    base_ring,
                    num_cache_pages,
                    table_width,
                )
        cute.arch.sync_threads()

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
        warp_mma_gemm(
            tiled_mma_pv,
            accumulator_output,
            register_probability,
            register_value,
            copy_source_probability,
            copy_source_value,
            copy_probability,
            copy_value,
        )
        cute.arch.sync_threads()
        if const_expr(self.split_qk):
            if is_resolver:
                key_base = Int64(-1)
                if (resolved_page >= Int64(0)) & (resolved_page < num_cache_pages):
                    key_base = (
                        resolved_page * Int64(self.key_page_stride)
                        + (resolved_position % Int64(self.page_size)) * Int64(self.key_token_stride)
                        + kv_head.to(Int64) * Int64(self.key_head_stride)
                    )
                base_ring[next_next_slot * Int32(self.tile_n) + resolve_column] = key_base
                position_ring[slot * Int32(self.tile_n) + resolve_column] = next_position
        self._store_fp8_gather(staged, shared_key, shared_value, thread)
        cute.arch.sync_threads()

    # ---- TMA (cp.async.bulk) gather -------------------------------------------------------
    @cute.jit
    def _segment_base_u32(self, tile_tensor: cute.Tensor, row: Int32, block: Int32) -> Int32:
        """Byte address of the 128-byte smem segment holding BF16 columns [64*block, 64*block+64) of row.

        The SW128 atom is 8 rows x 128 bytes, so each (row, block) owns one aligned 128-byte segment."""
        address = shared_ptr_to_u32(
            tile_tensor.iterator + cute.crd2idx((row, block * Int32(64)), tile_tensor.layout)
        )
        return address & Int32(~127)

    @cute.jit
    def _chunk_u32(self, tile_tensor: cute.Tensor, row: Int32, column: Int32) -> Int32:
        return shared_ptr_to_u32(
            tile_tensor.iterator + cute.crd2idx((row, column), tile_tensor.layout)
        )

    @cute.jit
    def _tma_issue_key(
        self,
        key_cache: cute.Pointer,
        base_ring: cute.Tensor,
        slot: Int32,
        lane: Int32,
        stage_u32: Int32,
        mbar_key: cute.Pointer,
    ) -> None:
        """One lane per row: bulk-copy the row's 256 FP8 bytes into the staging rows (Q smem)."""
        base = Int64(-1)
        if lane < Int32(self.tile_n):
            base = Int64(base_ring[slot * Int32(self.tile_n) + lane])
        tx = Int32(0)
        if base >= Int64(0):
            tx = Int32(_HEAD_DIM)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_key, tx)
        if base >= Int64(0):
            cp_async_bulk_g2s_mbar(
                stage_u32 + lane * Int32(_HEAD_DIM),
                _pointer_as_int64(key_cache, base),
                Int32(_HEAD_DIM),
                shared_ptr_to_u32(mbar_key),
            )

    @cute.jit
    def _tma_issue_value(
        self,
        value_cache: cute.Pointer,
        base_ring: cute.Tensor,
        slot: Int32,
        lane: Int32,
        shared_value: cute.Tensor,
        mbar_value: cute.Pointer,
    ) -> None:
        """One lane per row: four 64-byte bulk copies into the upper halves of the row's segments."""
        base = Int64(-1)
        if lane < Int32(self.tile_n):
            base = Int64(base_ring[slot * Int32(self.tile_n) + lane])
        tx = Int32(0)
        if base >= Int64(0):
            tx = Int32(_HEAD_DIM)
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_value, tx)
        if base >= Int64(0):
            for block in cutlass.range_constexpr(_HEAD_DIM // 64):
                cp_async_bulk_g2s_mbar(
                    self._segment_base_u32(shared_value, lane, Int32(block)) + Int32(64),
                    _pointer_as_int64(value_cache, base + Int64(block * 64)),
                    Int32(64),
                    shared_ptr_to_u32(mbar_value),
                )

    @cute.jit
    def _issue_value_prefetch(
        self, value_cache: cute.Pointer, base_ring: cute.Tensor, slot: Int32, thread: Int32,
        staged_v: cute.Tensor,
    ) -> None:
        """Register prefetch of one tile's V rows (the pipelined path, V only)."""
        vectors_per_row = _HEAD_DIM // 16
        for vector_iter in cutlass.range_constexpr(self.gather_iters):
            if const_expr(_ROWMAP):
                # Rows vary fastest across each 8-lane phase: with 16-byte-unit swizzle the chunk
                # is XORed with (row % 8), so 8 rows at one column hit 8 different bank groups.
                token = thread % Int32(8) + Int32(8 * vector_iter)
                dimension_base = (thread // Int32(8)) * Int32(16)
            else:
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
            base = Int64(base_ring[slot * Int32(self.tile_n) + token])
            v0 = Uint32(0)
            v1 = Uint32(0)
            v2 = Uint32(0)
            v3 = Uint32(0)
            if base >= Int64(0):
                v0, v1, v2, v3 = ld_global_nc_v4_u32(
                    _pointer_as_int64(value_cache, base + dimension_base.to(Int64))
                )
            staged_v[vector_iter * 4 + 0] = v0
            staged_v[vector_iter * 4 + 1] = v1
            staged_v[vector_iter * 4 + 2] = v2
            staged_v[vector_iter * 4 + 3] = v3

    @cute.jit
    def _store_value_prefetch(self, staged_v: cute.Tensor, shared_value: cute.Tensor, thread: Int32) -> None:
        vectors_per_row = _HEAD_DIM // 16
        for vector_iter in cutlass.range_constexpr(self.gather_iters):
            if const_expr(_ROWMAP):
                # Rows vary fastest across each 8-lane phase: with 16-byte-unit swizzle the chunk
                # is XORed with (row % 8), so 8 rows at one column hit 8 different bank groups.
                token = thread % Int32(8) + Int32(8 * vector_iter)
                dimension_base = (thread // Int32(8)) * Int32(16)
            else:
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
            self._store_fp8_one(
                shared_value, token, dimension_base,
                staged_v[vector_iter * 4 + 0], staged_v[vector_iter * 4 + 1],
                staged_v[vector_iter * 4 + 2], staged_v[vector_iter * 4 + 3],
            )

    @cute.jit
    def _write_bf16_pair(self, tile_tensor, row, column, x0, x1, x2, x3) -> None:
        """Convert 16 FP8 values (x0..x3) and store them as BF16 columns [column, column+16)."""
        y00, y01 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x0)
        y10, y11 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x1)
        y20, y21 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x2)
        y30, y31 = fp8x4_e4m3_to_bfloat2x2_native_sm120(x3)
        st_shared_v4_u32(self._chunk_u32(tile_tensor, row, column), y00, y01, y10, y11)
        st_shared_v4_u32(self._chunk_u32(tile_tensor, row, column + Int32(8)), y20, y21, y30, y31)

    @cute.jit
    def _segment_of(self, segment: Int32):
        """Map a segment id to (row, block): 8 consecutive ids = 8 rows of one atom, same block."""
        row = (segment % Int32(8)) + Int32(8) * (segment // Int32(32))
        block = (segment // Int32(8)) % Int32(4)
        return row, block

    @cute.jit
    def _tma_convert_key(
        self,
        base_ring: cute.Tensor,
        slot: Int32,
        thread: Int32,
        stage_u32: Int32,
        shared_key: cute.Tensor,
    ) -> None:
        """All threads: FP8 staging rows -> swizzled BF16 K tile. Rotated chunk order per lane keeps
        the 16-byte staging reads at most 2-way bank-conflicted; BF16 stores are conflict-free."""
        for it in cutlass.range_constexpr((self.segments + self.threads - 1) // self.threads):
            segment = thread + Int32(it * self.threads)
            if segment < Int32(self.segments):
                row, block = self._segment_of(segment)
                valid = Int64(base_ring[slot * Int32(self.tile_n) + row]) >= Int64(0)
                rotation = thread % Int32(4)
                for i in cutlass.range_constexpr(4):
                    q = (Int32(i) + rotation) % Int32(4)
                    column = block * Int32(64) + q * Int32(16)
                    x0 = Uint32(0)
                    x1 = Uint32(0)
                    x2 = Uint32(0)
                    x3 = Uint32(0)
                    if valid:
                        x0, x1, x2, x3 = ld_shared_v4_u32(
                            stage_u32 + row * Int32(_HEAD_DIM) + block * Int32(64) + q * Int32(16)
                        )
                    self._write_bf16_pair(shared_key, row, column, x0, x1, x2, x3)

    @cute.jit
    def _tma_convert_value(
        self,
        base_ring: cute.Tensor,
        slot: Int32,
        worker: Int32,
        workers: cutlass.Constexpr,
        shared_value: cute.Tensor,
    ) -> None:
        """Expand each segment's FP8 upper half to BF16 in place: read all four FP8 chunks first,
        then write the eight BF16 chunks (one thread owns a segment, so there is no cross-thread race)."""
        for it in cutlass.range_constexpr((self.segments + workers - 1) // workers):
            segment = worker + Int32(it * workers)
            if segment < Int32(self.segments):
                row, block = self._segment_of(segment)
                valid = Int64(base_ring[slot * Int32(self.tile_n) + row]) >= Int64(0)
                fp8_u32 = self._segment_base_u32(shared_value, row, block) + Int32(64)
                rotation = worker % Int32(4)
                staged = cute.make_rmem_tensor((16,), Uint32)
                for i in cutlass.range_constexpr(4):
                    q = (Int32(i) + rotation) % Int32(4)
                    x0 = Uint32(0)
                    x1 = Uint32(0)
                    x2 = Uint32(0)
                    x3 = Uint32(0)
                    if valid:
                        x0, x1, x2, x3 = ld_shared_v4_u32(fp8_u32 + q * Int32(16))
                    staged[i * 4 + 0] = x0
                    staged[i * 4 + 1] = x1
                    staged[i * 4 + 2] = x2
                    staged[i * 4 + 3] = x3
                for i in cutlass.range_constexpr(4):
                    q = (Int32(i) + rotation) % Int32(4)
                    self._write_bf16_pair(
                        shared_value, row, block * Int32(64) + q * Int32(16),
                        staged[i * 4 + 0], staged[i * 4 + 1], staged[i * 4 + 2], staged[i * 4 + 3],
                    )

    @cute.jit
    def _tma_step(
        self,
        local_tile: Int32,
        split: Int32,
        splits: Int32,
        row: Int32,
        kv_head: Int32,
        request_id: Int64,
        request_valid,
        query_position: Int64,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        base_ring: cute.Tensor,
        stage_u32: Int32,
        mbar_key: cute.Pointer,
        mbar_value: cute.Pointer,
        staged_v: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        shared_probability: cute.Tensor,
        shared_row_scale: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
        thread: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        register_query: cute.Tensor,
        register_key: cute.Tensor,
        register_probability: cute.Tensor,
        register_value: cute.Tensor,
        copy_query: cute.TiledCopy,
        copy_key: cute.TiledCopy,
        copy_probability: cute.TiledCopy,
        copy_value: cute.TiledCopy,
        copy_source_query: cute.Tensor,
        copy_source_key: cute.Tensor,
        copy_source_probability: cute.Tensor,
        copy_source_value: cute.Tensor,
        accumulator_output: cute.Tensor,
        softmax: Softmax,
        is_first: cutlass.Constexpr,
    ) -> None:
        """One TMA tile. Tile 0 (is_first) arrives via the synchronous prologue; tiles >= 1 arrive
        through two mbarriers whose n-th completion (n >= 1) is tile n, so wait parity = (n-1) & 1.

        a. (i>=1) all: wait K, convert staging -> K tile; sync
        b. (i>=1) copy warp: issue K(i+1) into the staging rows
        c. warp 0: QK(i), softmax, P -> smem.  warps 1..: (i>=1) wait V, expand V in place;
           resolver lanes: positions of tile i+2
        d. sync; (i==0) copy warp issues K(1) now that Q has left smem; all: PV(i); sync
        e. copy warp: issue V(i+1) into the V tile's upper halves
        """
        slot = local_tile % Int32(_RING)
        next_slot = (local_tile + Int32(1)) % Int32(_RING)
        next_next_slot = (local_tile + Int32(2)) % Int32(_RING)
        parity = (local_tile + Int32(1)) % Int32(2)  # == (local_tile - 1) & 1 for local_tile >= 1
        copy_warp = thread // Int32(32) == Int32(1)
        lane = thread % Int32(32)

        if const_expr(not is_first):
            cute.arch.mbarrier_wait(mbar_key, parity)
            self._tma_convert_key(base_ring, slot, thread, stage_u32, shared_key)
            cute.arch.sync_threads()
            if copy_warp:
                cute.arch.fence_proxy("async.shared", space="cta")
                self._tma_issue_key(key_cache, base_ring, next_slot, lane, stage_u32, mbar_key)
        if const_expr(not self.tma_value):
            self._issue_value_prefetch(value_cache, base_ring, next_slot, thread, staged_v)

        if thread < Int32(32):
            accumulator_score = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C((self.tile_m, self.tile_n)),
                Float32,
            )
            accumulator_score.fill(0.0)
            warp_mma_gemm(
                tiled_mma_qk,
                accumulator_score,
                register_query,
                register_key,
                copy_source_query,
                copy_source_key,
                copy_query,
                copy_key,
                A_in_regs=not is_first,
            )
            score_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    if (coordinate[0] >= Int32(self.heads_per_kv)) | (
                        Int64(base_ring[slot * Int32(self.tile_n) + coordinate[1]]) < Int64(0)
                    ):
                        score_mn[m, n] = -Float32.inf
            row_scale = softmax.online_softmax(accumulator_score, is_first=is_first, check_inf=True)
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    shared_probability[coordinate[0], coordinate[1]] = BFloat16(score_mn[m, n])
                if score_coordinates[m, 0][1] == Int32(0):
                    shared_row_scale[score_coordinates[m, 0][0]] = row_scale[m]
        else:
            if const_expr(not is_first and self.tma_value):
                cute.arch.mbarrier_wait(mbar_value, parity)
                self._tma_convert_value(
                    base_ring, slot, thread - Int32(32), self.threads - 32, shared_value
                )
            column = thread - Int32(self.resolver_offset)
            if column < Int32(self.tile_n):
                self._resolve_tile_positions(
                    split + (local_tile + Int32(2)) * splits,
                    next_next_slot,
                    column,
                    row,
                    kv_head,
                    request_id,
                    request_valid,
                    query_position,
                    block_table,
                    selected_positions,
                    base_ring,
                    num_cache_pages,
                    table_width,
                )
        cute.arch.sync_threads()

        if const_expr(is_first):
            if copy_warp:
                cute.arch.fence_proxy("async.shared", space="cta")
                self._tma_issue_key(key_cache, base_ring, next_slot, lane, stage_u32, mbar_key)

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
        warp_mma_gemm(
            tiled_mma_pv,
            accumulator_output,
            register_probability,
            register_value,
            copy_source_probability,
            copy_source_value,
            copy_probability,
            copy_value,
        )
        cute.arch.sync_threads()
        if const_expr(self.tma_value):
            if copy_warp:
                cute.arch.fence_proxy("async.shared", space="cta")
                self._tma_issue_value(value_cache, base_ring, next_slot, lane, shared_value, mbar_value)
        else:
            self._store_value_prefetch(staged_v, shared_value, thread)

    @cute.jit
    def _split_online_softmax(
        self,
        accumulator_score: cute.Tensor,
        softmax: Softmax,
        score_coordinates: cute.Tensor,
        reduce_max: cute.Tensor,
        thread: Int32,
        is_first: cutlass.Constexpr,
    ) -> cute.Tensor:
        """Online softmax when each warp holds tile_n/4 columns of the same 16 rows.

        Row maxima are exchanged through smem (one barrier); row sums stay per-thread partials
        over the thread's own columns (combined across warps once, in _combine_split_row_sums).
        The rescale factor depends only on the shared row max, so it is identical in every warp.
        Mirrors Softmax.online_softmax otherwise (same check_inf handling)."""
        warp_index = thread // Int32(32)
        lane = thread % Int32(32)
        acc_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
        rows = cute.size(acc_mn.shape[0])
        for r in cutlass.range_constexpr(rows):
            local_max = _cute_ops.fmax_reduce(acc_mn[r, None].load(), None, arch=120)
            local_max = cute.arch.warp_reduction_max(local_max, threads_in_group=4)
            if lane % Int32(4) == Int32(0):
                reduce_max[warp_index * Int32(self.tile_m) + score_coordinates[r, 0][0]] = local_max
        cute.arch.sync_threads()
        row_scale = cute.make_rmem_tensor(cute.make_layout((rows,), stride=(1,)), Float32)
        for r in cutlass.range_constexpr(rows):
            head = score_coordinates[r, 0][0]
            tile_max = Float32(reduce_max[head])
            for w in cutlass.range_constexpr(1, 4):
                tile_max = fmax(tile_max, Float32(reduce_max[Int32(w * self.tile_m) + head]))
            row_max_prev = softmax.row_max[r]
            row_max_cur = tile_max
            if const_expr(not is_first):
                row_max_cur = fmax(row_max_prev, tile_max)
            softmax.row_max[r] = row_max_cur
            row_max_cur = Float32(cutlass.select_(row_max_cur == -Float32.inf, 0.0, row_max_cur))
            acc_row = acc_mn[r, None].load()
            acc_row_exp = cute.math.exp2(
                acc_row * softmax.scale_log2 - row_max_cur * softmax.scale_log2, fastmath=True
            )
            if const_expr(is_first):
                row_scale[r] = 1.0
                softmax.row_sum[r] = _cute_ops.fadd_reduce(acc_row_exp, None, arch=120)
            else:
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * softmax.scale_log2, fastmath=True
                )
                softmax.row_sum[r] = _cute_ops.fadd_reduce(
                    acc_row_exp, softmax.row_sum[r] * row_scale[r], arch=120
                )
            acc_mn[r, None].store(acc_row_exp)
        return row_scale

    @cute.jit
    def _combine_split_row_sums(
        self,
        softmax: Softmax,
        score_coordinates: cute.Tensor,
        reduce_sum: cute.Tensor,
        thread: Int32,
    ) -> None:
        """Sum per-warp partial row sums so that warp 0's quad lane 0 holds the total
        (other lanes 0), which Softmax.finalize then quad-reduces unchanged."""
        warp_index = thread // Int32(32)
        lane = thread % Int32(32)
        rows = cute.size(softmax.row_sum)
        for r in cutlass.range_constexpr(rows):
            total = softmax.row_sum[r]
            total += cute.arch.shuffle_sync_bfly(total, offset=2)
            total += cute.arch.shuffle_sync_bfly(total, offset=1)
            if lane % Int32(4) == Int32(0):
                reduce_sum[warp_index * Int32(self.tile_m) + score_coordinates[r, 0][0]] = total
        cute.arch.sync_threads()
        if warp_index == Int32(0):
            for r in cutlass.range_constexpr(rows):
                head = score_coordinates[r, 0][0]
                total = Float32(reduce_sum[head])
                for w in cutlass.range_constexpr(1, 4):
                    total += Float32(reduce_sum[Int32(w * self.tile_m) + head])
                softmax.row_sum[r] = Float32(cutlass.select_(lane % Int32(4) == Int32(0), total, 0.0))

    def _shared_layouts(
        self,
    ) -> tuple[
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
        cute.ComposedLayout,
    ]:
        head_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                _HEAD_DIM,
            ),
            BFloat16,
        )
        if const_expr(_SWZ333):
            head_layout_atom = cute.make_composed_layout(
                cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1))
            )
        probability_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                self.tile_n,
            ),
            BFloat16,
        )
        if const_expr(_SWZ333):
            swizzle_bits = {16: 1, 32: 2, 64: 3}[self.tile_n]
            probability_layout_atom = cute.make_composed_layout(
                cute.make_swizzle(swizzle_bits, 3, 3),
                0,
                cute.make_layout((8, self.tile_n), stride=(self.tile_n, 1)),
            )
        query_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_m, _HEAD_DIM),
            order=(0, 1),
        )
        key_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, _HEAD_DIM),
            order=(0, 1),
        )
        value_layout = cute.tile_to_shape(
            head_layout_atom,
            (self.tile_n, _HEAD_DIM),
            order=(0, 1),
        )
        probability_layout = cute.tile_to_shape(
            probability_layout_atom,
            (self.tile_m, self.tile_n),
            order=(0, 1),
        )
        return query_layout, key_layout, value_layout, probability_layout

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        query_layout, key_layout, value_layout, probability_layout = (
            self._shared_layouts()
        )
        tiled_mma_qk = self._tiled_mma_qk()
        tiled_mma_pv = self._tiled_mma_pv()
        self.kernel(
            query,
            key_cache,
            value_cache,
            k_descale,
            v_descale,
            block_table,
            request_ids,
            selected_positions,
            query_positions,
            partial_output,
            partial_lse,
            output,
            num_cache_pages,
            table_batch,
            table_width,
            softmax_scale,
            splits,
            query_layout,
            key_layout,
            value_layout,
            probability_layout,
            tiled_mma_qk,
            tiled_mma_pv,
        ).launch(
            grid=(rows, self.kv_heads, splits),
            block=(self.threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _mma_selected_tile(
        self,
        tile: Int32,
        row: Int32,
        kv_head: Int32,
        request_id: Int64,
        request_valid,
        query_position: Int64,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        block_table: cute.Pointer,
        selected_positions: cute.Pointer,
        key_bases: cute.Tensor,
        value_bases: cute.Tensor,
        shared_query: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        shared_probability: cute.Tensor,
        shared_row_scale: cute.Tensor,
        num_cache_pages: Int64,
        table_width: Int64,
        thread: Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        register_query: cute.Tensor,
        register_key: cute.Tensor,
        register_probability: cute.Tensor,
        register_value: cute.Tensor,
        copy_query: cute.TiledCopy,
        copy_key: cute.TiledCopy,
        copy_probability: cute.TiledCopy,
        copy_value: cute.TiledCopy,
        copy_source_query: cute.Tensor,
        copy_source_key: cute.Tensor,
        copy_source_probability: cute.Tensor,
        copy_source_value: cute.Tensor,
        accumulator_output: cute.Tensor,
        softmax: Softmax,
        is_first: cutlass.Constexpr,
    ) -> None:
        column = thread
        if column < Int32(self.tile_n):
            selected_column = tile * Int32(self.tile_n) + column
            valid = request_valid & (selected_column < Int32(self.selection_width))
            logical_position = Int64(-1)
            if valid:
                selected_offset = row.to(Int64) * Int64(
                    self.selection_width
                ) + selected_column.to(Int64)
                logical_position = selected_positions[selected_offset].to(Int64)
                valid = (logical_position >= Int64(0)) & (
                    logical_position <= query_position
                )

            physical_page = Int64(-1)
            page_offset = Int64(0)
            if valid:
                logical_page = logical_position // Int64(self.page_size)
                valid = logical_page < table_width
                if valid:
                    table_offset = request_id * table_width + logical_page
                    physical_page = block_table[table_offset].to(Int64)
                    valid = (physical_page >= Int64(0)) & (
                        physical_page < num_cache_pages
                    )
                    page_offset = logical_position % Int64(self.page_size)

            key_base = Int64(-1)
            value_base = Int64(-1)
            if valid:
                key_base = (
                    physical_page * Int64(self.key_page_stride)
                    + page_offset * Int64(self.key_token_stride)
                    + kv_head.to(Int64) * Int64(self.key_head_stride)
                )
                if const_expr(self.shared_cache_strides):
                    value_base = key_base
                else:
                    value_base = (
                        physical_page * Int64(self.value_page_stride)
                        + page_offset * Int64(self.value_token_stride)
                        + kv_head.to(Int64) * Int64(self.value_head_stride)
                    )
            key_bases[column] = key_base
            if const_expr(not self.shared_cache_strides):
                value_bases[column] = value_base
        cute.arch.sync_threads()

        if const_expr(not self.kv_is_fp8):
            vectors_per_row = _HEAD_DIM // 8
            for vector_iter in cutlass.range_constexpr(
                self.tile_n * vectors_per_row // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                vector = linear_vector % Int32(vectors_per_row)
                dimension_base = vector * Int32(8)
                key_base = Int64(key_bases[token])
                value_base = key_base
                if const_expr(not self.shared_cache_strides):
                    value_base = Int64(value_bases[token])
                key_valid = key_base >= Int64(0)
                value_valid = value_base >= Int64(0)
                safe_key_base = Int64(cutlass.select_(key_valid, key_base, Int64(0)))
                safe_value_base = Int64(
                    cutlass.select_(value_valid, value_base, Int64(0))
                )
                key_destination = shared_ptr_to_u32(
                    shared_key.iterator
                    + cute.crd2idx(
                        (token, dimension_base),
                        shared_key.layout,
                    )
                )
                value_destination = shared_ptr_to_u32(
                    shared_value.iterator
                    + cute.crd2idx(
                        (token, dimension_base),
                        shared_value.layout,
                    )
                )
                _cp_async_load_128b_zfill(
                    key_destination,
                    _pointer_as_int64(
                        key_cache,
                        safe_key_base + dimension_base.to(Int64),
                    ),
                    cutlass.select_(key_valid, Int32(16), Int32(0)),
                )
                _cp_async_load_128b_zfill(
                    value_destination,
                    _pointer_as_int64(
                        value_cache,
                        safe_value_base + dimension_base.to(Int64),
                    ),
                    cutlass.select_(value_valid, Int32(16), Int32(0)),
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        else:
            vectors_per_row = _HEAD_DIM // 16
            for vector_iter in cutlass.range_constexpr(
                self.tile_n * vectors_per_row // self.threads
            ):
                linear_vector = thread + Int32(vector_iter * self.threads)
                token = linear_vector // Int32(vectors_per_row)
                dimension_base = (linear_vector % Int32(vectors_per_row)) * Int32(16)
                key_base = Int64(key_bases[token])
                value_base = key_base
                if const_expr(not self.shared_cache_strides):
                    value_base = Int64(value_bases[token])
                self._load_fp8_vector_to_bf16_shared(
                    key_cache,
                    key_base,
                    shared_key,
                    token,
                    dimension_base,
                )
                self._load_fp8_vector_to_bf16_shared(
                    value_cache,
                    value_base,
                    shared_value,
                    token,
                    dimension_base,
                )
        cute.arch.sync_threads()

        if thread < Int32(self.qk_warps * 32):
            accumulator_score = cute.make_rmem_tensor(
                tiled_mma_qk.get_slice(thread).partition_shape_C(
                    (self.tile_m, self.tile_n)
                ),
                Float32,
            )
            accumulator_score.fill(0.0)
            warp_mma_gemm(
                tiled_mma_qk,
                accumulator_score,
                register_query,
                register_key,
                copy_source_query,
                copy_source_key,
                copy_query,
                copy_key,
                A_in_regs=not is_first,
            )
            score_mn = layout_utils.reshape_acc_to_mn(accumulator_score)
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    if (coordinate[0] >= Int32(self.heads_per_kv)) | (
                        Int64(key_bases[coordinate[1]]) < Int64(0)
                    ):
                        score_mn[m, n] = -Float32.inf

            row_scale = softmax.online_softmax(
                accumulator_score,
                is_first=is_first,
                check_inf=True,
            )
            for m in cutlass.range_constexpr(cute.size(score_mn.shape[0])):
                for n in cutlass.range_constexpr(cute.size(score_mn.shape[1])):
                    coordinate = score_coordinates[m, n]
                    shared_probability[coordinate[0], coordinate[1]] = BFloat16(
                        score_mn[m, n]
                    )
                if score_coordinates[m, 0][1] == Int32(0):
                    shared_row_scale[score_coordinates[m, 0][0]] = row_scale[m]

        cute.arch.sync_threads()

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        if const_expr(not is_first):
            for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
                scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
                output_mn[m, None].store(output_mn[m, None].load() * scale)
        warp_mma_gemm(
            tiled_mma_pv,
            accumulator_output,
            register_probability,
            register_value,
            copy_source_probability,
            copy_source_value,
            copy_probability,
            copy_value,
        )
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        query: cute.Pointer,
        key_cache: cute.Pointer,
        value_cache: cute.Pointer,
        k_descale: cute.Pointer,
        v_descale: cute.Pointer,
        block_table: cute.Pointer,
        request_ids: cute.Pointer,
        selected_positions: cute.Pointer,
        query_positions: cute.Pointer,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        num_cache_pages: Int64,
        table_batch: Int64,
        table_width: Int64,
        softmax_scale: Float32,
        splits: Int32,
        query_layout: cute.ComposedLayout,
        key_layout: cute.ComposedLayout,
        value_layout: cute.ComposedLayout,
        probability_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
    ) -> None:
        row_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        thread = Int32(thread_idx)
        split = Int32(split_idx)
        kv_head = Int32(kv_head_idx)
        row = Int32(row_idx)
        request_id = request_ids[row].to(Int64)
        query_position = query_positions[row].to(Int64)
        request_valid = (request_id >= Int64(0)) & (request_id < table_batch)

        k_scale = Float32(1.0)
        v_scale = Float32(1.0)
        if const_expr(self.kv_is_fp8):
            k_scale = Float32(k_descale[0])
            v_scale = Float32(v_descale[0])

        allocator = cutlass.utils.SmemAllocator()
        shared_query = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=query_layout,
            byte_alignment=1024,
        )
        shared_key = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=key_layout,
            byte_alignment=1024,
        )
        shared_value = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=value_layout,
            byte_alignment=1024,
        )
        shared_probability = allocator.allocate_tensor(
            element_type=BFloat16,
            layout=probability_layout,
            byte_alignment=1024,
        )
        shared_row_scale = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((self.tile_m,), stride=(1,)),
            byte_alignment=16,
        )
        key_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )
        value_bases = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )
        base_ring = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((_RING * self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )
        mbarriers = allocator.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout((2,), stride=(1,)),
            byte_alignment=8,
        )
        position_ring = allocator.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout((_RING * self.tile_n,), stride=(1,)),
            byte_alignment=16,
        )
        reduce_max = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((4 * self.tile_m,), stride=(1,)),
            byte_alignment=16,
        )
        reduce_sum = allocator.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((4 * self.tile_m,), stride=(1,)),
            byte_alignment=16,
        )

        query_element = thread
        first_query_head = kv_head * Int32(self.heads_per_kv)
        while query_element < Int32(self.tile_m * _HEAD_DIM):
            local_head = query_element // Int32(_HEAD_DIM)
            dimension = query_element - local_head * Int32(_HEAD_DIM)
            query_value = Float32(0.0)
            if local_head < Int32(self.heads_per_kv):
                query_head = first_query_head + local_head
                query_offset = (
                    row.to(Int64) * Int64(self.q_heads) + query_head.to(Int64)
                ) * Int64(_HEAD_DIM) + dimension.to(Int64)
                query_value = Float32(query[query_offset])
            shared_query[local_head, dimension] = BFloat16(query_value)
            query_element += Int32(self.threads)
        cute.arch.sync_threads()

        qk_thread = thread % Int32(self.qk_threads)
        thread_mma_qk = tiled_mma_qk.get_slice(qk_thread)
        thread_mma_pv = tiled_mma_pv.get_slice(thread)
        register_query = thread_mma_qk.make_fragment_A(
            thread_mma_qk.partition_A(shared_query)
        )
        register_key = thread_mma_qk.make_fragment_B(
            thread_mma_qk.partition_B(shared_key)
        )
        shared_value_transposed = layout_utils.transpose_view(shared_value)
        register_probability = thread_mma_pv.make_fragment_A(
            thread_mma_pv.partition_A(shared_probability)
        )
        register_value = thread_mma_pv.make_fragment_B(
            thread_mma_pv.partition_B(shared_value_transposed)
        )
        copy_atom_query_key = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            BFloat16,
        )
        copy_atom_value = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            BFloat16,
        )
        copy_query = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_qk
        ).get_slice(qk_thread)
        copy_key = cute.make_tiled_copy_B(copy_atom_query_key, tiled_mma_qk).get_slice(
            qk_thread
        )
        copy_probability = cute.make_tiled_copy_A(
            copy_atom_query_key, tiled_mma_pv
        ).get_slice(thread)
        copy_value = cute.make_tiled_copy_B(copy_atom_value, tiled_mma_pv).get_slice(
            thread
        )
        copy_source_query = copy_query.partition_S(shared_query)
        copy_source_key = copy_key.partition_S(shared_key)
        copy_source_probability = copy_probability.partition_S(shared_probability)
        copy_source_value = copy_value.partition_S(shared_value_transposed)
        accumulator_output = cute.make_rmem_tensor(
            thread_mma_pv.partition_shape_C((self.tile_m, _HEAD_DIM)),
            Float32,
        )
        accumulator_output.fill(0.0)
        score_layout = thread_mma_qk.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n))
        ).layout
        score_mn_layout = layout_utils.convert_layout_acc_mn(score_layout)
        score_scale = Float32(softmax_scale * Float32(_LOG2_E))
        if const_expr(self.kv_is_fp8):
            score_scale *= k_scale
        softmax = Softmax.create(
            score_scale,
            cute.size(score_mn_layout.shape[0]),
            arch=120,
        )
        softmax.reset()

        tile_count = Int32((self.selection_width + self.tile_n - 1) // self.tile_n)
        tiles_per_split = (tile_count - split + splits - Int32(1)) // splits
        if const_expr(self.tma):
            mbar_key = mbarriers.iterator
            mbar_value = mbarriers.iterator + 1
            stage_u32 = shared_ptr_to_u32(shared_query.iterator)
            if thread == Int32(0):
                cute.arch.mbarrier_init(mbar_key, Int32(32))
                cute.arch.mbarrier_init(mbar_value, Int32(32))
                cute.arch.mbarrier_init_fence()
            # Prologue: resolve tiles 0 and 1; tile 0 through the synchronous gather.
            if thread < Int32(self.tile_n):
                self._resolve_tile_positions(
                    split, Int32(0), thread, row, kv_head, request_id, request_valid,
                    query_position, block_table, selected_positions, base_ring,
                    num_cache_pages, table_width,
                )
                self._resolve_tile_positions(
                    split + splits, Int32(1), thread, row, kv_head, request_id, request_valid,
                    query_position, block_table, selected_positions, base_ring,
                    num_cache_pages, table_width,
                )
            cute.arch.sync_threads()
            staged0 = cute.make_rmem_tensor((self.gather_iters * 8,), Uint32)
            staged_v = cute.make_rmem_tensor((self.gather_iters * 4,), Uint32)
            self._issue_fp8_gather(key_cache, value_cache, base_ring, Int32(0), thread, staged0)
            self._store_fp8_gather(staged0, shared_key, shared_value, thread)
            cute.arch.sync_threads()
            self._tma_step(
                Int32(0), split, splits, row, kv_head, request_id, request_valid, query_position,
                key_cache, value_cache, block_table, selected_positions, base_ring, stage_u32,
                mbar_key, mbar_value, staged_v, shared_key, shared_value, shared_probability,
                shared_row_scale, num_cache_pages, table_width, thread, tiled_mma_qk,
                tiled_mma_pv, register_query, register_key, register_probability, register_value,
                copy_query, copy_key, copy_probability, copy_value, copy_source_query,
                copy_source_key, copy_source_probability, copy_source_value, accumulator_output,
                softmax, is_first=True,
            )
            for local_tile in cutlass.range(tiles_per_split - Int32(1), unroll=1):
                self._tma_step(
                    Int32(local_tile) + Int32(1), split, splits, row, kv_head, request_id,
                    request_valid, query_position, key_cache, value_cache, block_table,
                    selected_positions, base_ring, stage_u32, mbar_key, mbar_value, staged_v, shared_key,
                    shared_value, shared_probability, shared_row_scale, num_cache_pages,
                    table_width, thread, tiled_mma_qk, tiled_mma_pv, register_query, register_key,
                    register_probability, register_value, copy_query, copy_key, copy_probability,
                    copy_value, copy_source_query, copy_source_key, copy_source_probability,
                    copy_source_value, accumulator_output, softmax, is_first=False,
                )
            # Drain: the copies issued for tile tiles_per_split cover only invalid rows (0 bytes),
            # but wait on both barriers so no async write can land after the CTA moves on.
            cute.arch.mbarrier_wait(mbar_key, (tiles_per_split + Int32(1)) % Int32(2))
            if const_expr(self.tma_value):
                cute.arch.mbarrier_wait(mbar_value, (tiles_per_split + Int32(1)) % Int32(2))
        elif const_expr(self.pipelined):
            staged = cute.make_rmem_tensor((self.gather_iters * 8,), Uint32)
            # Prologue: resolve tiles 0 and 1, gather + store tile 0.
            if thread < Int32(self.tile_n):
                self._resolve_tile_positions(
                    split, Int32(0), thread, row, kv_head, request_id, request_valid,
                    query_position, block_table, selected_positions, base_ring,
                    num_cache_pages, table_width,
                )
                self._resolve_tile_positions(
                    split + splits, Int32(1), thread, row, kv_head, request_id, request_valid,
                    query_position, block_table, selected_positions, base_ring,
                    num_cache_pages, table_width,
                )
                if const_expr(self.split_qk):
                    # Park tile 2's raw positions for the deferred resolver (slot 2).
                    selected_column = (split + Int32(2) * splits) * Int32(self.tile_n) + thread
                    parked = Int32(-1)
                    if request_valid & (selected_column < Int32(self.selection_width)):
                        parked = selected_positions[
                            row.to(Int64) * Int64(self.selection_width) + selected_column.to(Int64)
                        ]
                    position_ring[Int32(2 * self.tile_n) + thread] = parked
            cute.arch.sync_threads()
            self._issue_fp8_gather(key_cache, value_cache, base_ring, Int32(0), thread, staged)
            self._store_fp8_gather(staged, shared_key, shared_value, thread)
            cute.arch.sync_threads()
            self._pipelined_step(
                Int32(0), split, splits, row, kv_head, request_id, request_valid, query_position,
                key_cache, value_cache, block_table, selected_positions, base_ring, staged,
                reduce_max, position_ring, shared_key, shared_value, shared_probability, shared_row_scale,
                num_cache_pages, table_width, thread, tiled_mma_qk, tiled_mma_pv,
                register_query, register_key, register_probability, register_value,
                copy_query, copy_key, copy_probability, copy_value,
                copy_source_query, copy_source_key, copy_source_probability, copy_source_value,
                accumulator_output, softmax, is_first=True,
            )
            for local_tile in cutlass.range(tiles_per_split - Int32(1), unroll=1):
                self._pipelined_step(
                    Int32(local_tile) + Int32(1), split, splits, row, kv_head, request_id,
                    request_valid, query_position, key_cache, value_cache, block_table,
                    selected_positions, base_ring, staged, reduce_max, position_ring, shared_key, shared_value,
                    shared_probability, shared_row_scale, num_cache_pages, table_width, thread,
                    tiled_mma_qk, tiled_mma_pv, register_query, register_key,
                    register_probability, register_value, copy_query, copy_key,
                    copy_probability, copy_value, copy_source_query, copy_source_key,
                    copy_source_probability, copy_source_value, accumulator_output, softmax,
                    is_first=False,
                )
        else:
            self._mma_selected_tile(
                split,
                row,
                kv_head,
                request_id,
                request_valid,
                query_position,
                key_cache,
                value_cache,
                block_table,
                selected_positions,
                key_bases,
                value_bases,
                shared_query,
                shared_key,
                shared_value,
                shared_probability,
                shared_row_scale,
                num_cache_pages,
                table_width,
                thread,
                tiled_mma_qk,
                tiled_mma_pv,
                register_query,
                register_key,
                register_probability,
                register_value,
                copy_query,
                copy_key,
                copy_probability,
                copy_value,
                copy_source_query,
                copy_source_key,
                copy_source_probability,
                copy_source_value,
                accumulator_output,
                softmax,
                is_first=True,
            )
            for local_tile in cutlass.range(tiles_per_split - Int32(1), unroll=1):
                tile = split + (Int32(local_tile) + Int32(1)) * splits
                self._mma_selected_tile(
                    tile,
                    row,
                    kv_head,
                    request_id,
                    request_valid,
                    query_position,
                    key_cache,
                    value_cache,
                    block_table,
                    selected_positions,
                    key_bases,
                    value_bases,
                    shared_query,
                    shared_key,
                    shared_value,
                    shared_probability,
                    shared_row_scale,
                    num_cache_pages,
                    table_width,
                    thread,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    register_query,
                    register_key,
                    register_probability,
                    register_value,
                    copy_query,
                    copy_key,
                    copy_probability,
                    copy_value,
                    copy_source_query,
                    copy_source_key,
                    copy_source_probability,
                    copy_source_value,
                    accumulator_output,
                    softmax,
                    is_first=False,
                )

        if const_expr(self.split_qk):
            self._combine_split_row_sums(
                softmax,
                layout_utils.reshape_acc_to_mn(
                    tiled_mma_qk.get_slice(thread).partition_C(
                        cute.make_identity_tensor((self.tile_m, self.tile_n))
                    )
                ),
                reduce_sum,
                thread,
            )
        if thread < Int32(self.qk_warps * 32):
            output_scale = softmax.finalize()
            score_coordinates = layout_utils.reshape_acc_to_mn(
                tiled_mma_qk.get_slice(qk_thread).partition_C(
                    cute.make_identity_tensor((self.tile_m, self.tile_n))
                )
            )
            if score_coordinates[0, 0][1] == Int32(0):
                for m in cutlass.range_constexpr(cute.size(softmax.row_sum)):
                    local_head = score_coordinates[m, 0][0]
                    scale = output_scale[m]
                    if const_expr(self.kv_is_fp8):
                        scale *= v_scale
                    shared_row_scale[local_head] = scale
                    if const_expr(
                        not self.direct_output or self.return_lse
                    ) and local_head < Int32(self.heads_per_kv):
                        query_head = first_query_head + local_head
                        lse_offset = (
                            row.to(Int64) * splits.to(Int64) + split.to(Int64)
                        ) * Int64(self.q_heads) + query_head.to(Int64)
                        partial_lse[lse_offset] = Float32(softmax.row_sum[m])
        cute.arch.sync_threads()

        output_mn = layout_utils.reshape_acc_to_mn(accumulator_output)
        output_coordinates = layout_utils.reshape_acc_to_mn(
            tiled_mma_pv.get_slice(thread).partition_C(
                cute.make_identity_tensor((self.tile_m, _HEAD_DIM))
            )
        )
        output_base = (
            row.to(Int64) * Int64(self.q_heads)
            if const_expr(self.direct_output)
            else (row.to(Int64) * splits.to(Int64) + split.to(Int64))
            * Int64(self.q_heads)
        ) * Int64(_HEAD_DIM)
        for m in cutlass.range_constexpr(cute.size(output_mn.shape[0])):
            scale = Float32(shared_row_scale[output_coordinates[m, 0][0]])
            output_mn[m, None].store(output_mn[m, None].load() * scale)
            for n in cutlass.range_constexpr(cute.size(output_mn.shape[1])):
                coordinate = output_coordinates[m, n]
                if coordinate[0] < Int32(self.heads_per_kv):
                    query_head = first_query_head + coordinate[0]
                    offset = (
                        output_base
                        + query_head.to(Int64) * Int64(_HEAD_DIM)
                        + coordinate[1].to(Int64)
                    )
                    if const_expr(self.direct_output):
                        output[offset] = BFloat16(output_mn[m, n])
                    else:
                        partial_output[offset] = Float32(output_mn[m, n])


class _SparseGqaMergeKernel:
    """Merge FP32 split-softmax partials into caller-owned BF16 output."""

    def __init__(self, *, q_heads: int, return_lse: bool) -> None:
        self.q_heads = int(q_heads)
        self.return_lse = bool(return_lse)

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, output_lse, splits).launch(
            grid=(self.q_heads, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        thread_i = Int32(thread)
        row_i = Int64(row)
        head_i = Int64(query_head)
        lane = thread_i % Int32(32)
        warp_id = thread_i // Int32(32)
        weights = cute.make_rmem_tensor((_NUM_SPLITS // 32,), Float32)
        local_maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            local_maximum = fmax(local_maximum, weight)
        maximum = warp_reduce(local_maximum, fmax)

        local_denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            local_denominator += weight
        denominator = warp_reduce(local_denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator
        if const_expr(self.return_lse) and thread_i == Int32(0):
            merged_lse = Float32(-Float32.inf)
            if denominator > Float32(0.0):
                merged_lse = maximum + cute.math.log2(
                    denominator, fastmath=True
                ) * Float32(0.6931471805599453)
            output_lse[row_i * Int64(self.q_heads) + head_i] = merged_lse

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((_NUM_SPLITS,), stride=(1,)),
            byte_alignment=16,
        )
        if warp_id == Int32(0):
            for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        # Adjacent lanes load adjacent head dimensions. Preserve the original
        # split-pair accumulation and ascending butterfly reduction order.
        for tile in cutlass.range_constexpr(_HEAD_DIM // _THREADS):
            dimension = thread_i + Int32(tile * _THREADS)
            partials = cute.make_rmem_tensor((32,), Float32)
            for split_lane in cutlass.range_constexpr(32):
                total = Float32(0.0)
                for slot in cutlass.range_constexpr(_NUM_SPLITS // 32):
                    split = Int32(split_lane + slot * 32)
                    if split < splits:
                        offset = (
                            (row_i * splits.to(Int64) + split.to(Int64))
                            * Int64(self.q_heads)
                            + head_i
                        ) * Int64(_HEAD_DIM) + dimension.to(Int64)
                        total += Float32(partial_output[offset]) * Float32(
                            shared_weights[split]
                        )
                partials[split_lane] = total
            for stage in cutlass.range_constexpr(5):
                for group in cutlass.range_constexpr(32 // (2 << stage)):
                    left = group * (2 << stage)
                    right = left + (1 << stage)
                    partials[left] = Float32(partials[left]) + Float32(partials[right])
            output_offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                _HEAD_DIM
            ) + dimension.to(Int64)
            output[output_offset] = BFloat16(Float32(partials[0]) * inverse)


class _CooperativeSparseGqaMergeKernel(_SparseGqaMergeKernel):
    """Distribute each ordered split reduction across four warps and dimension tiles."""

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(partial_output, partial_lse, output, output_lse, splits).launch(
            grid=(self.q_heads, rows, _HEAD_DIM // 32),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        splits: Int32,
    ) -> None:
        query_head, row, tile = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = Int32(thread) % Int32(32)
        warp = Int32(thread) // Int32(32)
        row_i, head_i = Int64(row), Int64(query_head)
        weights = cute.make_rmem_tensor((2,), Float32)
        maximum = Float32(-Float32.inf)
        for slot in cutlass.range_constexpr(2):
            split = lane + Int32(slot * 32)
            weight = Float32(-Float32.inf)
            if split < splits:
                offset = (row_i * splits.to(Int64) + split.to(Int64)) * Int64(
                    self.q_heads
                ) + head_i
                weight = Float32(partial_lse[offset])
            weights[slot] = weight
            maximum = fmax(maximum, weight)
        maximum = warp_reduce(maximum, fmax)
        denominator = Float32(0.0)
        for slot in cutlass.range_constexpr(2):
            weight = Float32(0.0)
            if weights[slot] > Float32(-Float32.inf):
                weight = _exp2_approx(
                    (Float32(weights[slot]) - maximum) * Float32(_LOG2_E)
                )
            weights[slot] = weight
            denominator += weight
        denominator = warp_reduce(denominator, _add)
        inverse = Float32(0.0)
        if denominator > Float32(0.0):
            inverse = Float32(1.0) / denominator
        if const_expr(self.return_lse) and (warp == Int32(0)) & (
            lane == Int32(0)
        ):
            merged_lse = Float32(-Float32.inf)
            if denominator > Float32(0.0):
                merged_lse = maximum + cute.math.log2(
                    denominator, fastmath=True
                ) * Float32(0.6931471805599453)
            output_lse[row_i * Int64(self.q_heads) + head_i] = merged_lse

        allocator = cutlass.utils.SmemAllocator()
        shared_weights = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((64,), stride=(1,)),
            byte_alignment=16,
        )
        shared_partials = allocator.allocate_tensor(
            Float32,
            layout=cute.make_layout((4, 32), stride=(32, 1)),
            byte_alignment=16,
        )
        if warp == Int32(0):
            for slot in cutlass.range_constexpr(2):
                shared_weights[lane + Int32(slot * 32)] = weights[slot]
        cute.arch.sync_threads()

        dimension = Int64(tile) * Int64(32) + lane.to(Int64)
        partials = cute.make_rmem_tensor((8,), Float32)
        for local in cutlass.range_constexpr(8):
            total = Float32(0.0)
            for slot in cutlass.range_constexpr(2):
                split = warp * Int32(8) + Int32(local + slot * 32)
                if split < splits:
                    offset = (
                        (row_i * splits.to(Int64) + split.to(Int64))
                        * Int64(self.q_heads)
                        + head_i
                    ) * Int64(_HEAD_DIM) + dimension
                    total += Float32(partial_output[offset]) * Float32(
                        shared_weights[split]
                    )
            partials[local] = total
        # Pair s with s+32; ascending butterfly stages 0..2 stay warp-local.
        for stage in cutlass.range_constexpr(3):
            for group in cutlass.range_constexpr(8 // (2 << stage)):
                left = group * (2 << stage)
                partials[left] = Float32(partials[left]) + Float32(
                    partials[left + (1 << stage)]
                )
        shared_partials[warp, lane] = partials[0]
        cute.arch.sync_threads()
        if warp == Int32(0):
            # Stages 3 and 4 join the four warp-owned contiguous groups.
            column = lane
            low = Float32(shared_partials[0, column]) + Float32(
                shared_partials[1, column]
            )
            high = Float32(shared_partials[2, column]) + Float32(
                shared_partials[3, column]
            )
            dimension = Int64(tile) * Int64(32) + column.to(Int64)
            offset = (row_i * Int64(self.q_heads) + head_i) * Int64(
                _HEAD_DIM
            ) + dimension
            output[offset] = BFloat16((low + high) * inverse)


class _ShapeAdaptiveSparseGqaMergeKernel:
    """Trade split parallelism for independent query rows in one compiled callable."""

    def __init__(self, *, q_heads: int, return_lse: bool) -> None:
        self.q_heads = q_heads
        self.return_lse = bool(return_lse)
        self.serial = _SparseGqaMergeKernel(
            q_heads=q_heads, return_lse=return_lse
        )
        self.cooperative = _CooperativeSparseGqaMergeKernel(
            q_heads=q_heads, return_lse=return_lse
        )

    @cute.jit
    def __call__(
        self,
        partial_output: cute.Pointer,
        partial_lse: cute.Pointer,
        output: cute.Pointer,
        output_lse: cute.Pointer,
        rows: Int32,
        splits: Int32,
        stream: cuda.CUstream,
    ) -> None:
        # Independent query rows supply parallelism without multiplying CTAs
        # across head dimensions. Both branches preserve the same reduction tree.
        if rows > splits:
            self.serial.kernel(
                partial_output, partial_lse, output, output_lse, splits
            ).launch(
                grid=(self.q_heads, rows, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
            )
        else:
            self.cooperative.kernel(
                partial_output, partial_lse, output, output_lse, splits
            ).launch(
                grid=(self.q_heads, rows, _HEAD_DIM // 32),
                block=(_THREADS, 1, 1),
                stream=stream,
            )


_MERGE_KERNEL = _ShapeAdaptiveSparseGqaMergeKernel


def _cache_key(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    *,
    direct_output: bool,
    return_lse: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
) -> tuple[object, ...]:
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        int(device_index),
        int(query.shape[1]),
        int(key_cache.shape[2]),
        int(request_ids.element_size() * 8),
        key_cache.dtype,
        int(key_cache.shape[1]),
        tuple(map(int, key_cache.stride()[:3])),
        tuple(map(int, value_cache.stride()[:3])),
        bool(direct_output),
        bool(return_lse) if direct_output else True,
        int(kv_warps),
        int(selection_width),
    )


def _pointer(
    tensor: torch.Tensor,
    dtype: type[cutlass.Numeric],
) -> cute.Pointer:
    return make_ptr(
        dtype,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _fake_pointer(dtype: type[cutlass.Numeric]) -> cute.Pointer:
    return make_ptr(
        dtype,
        16,
        cute.AddressSpace.gmem,
        assumed_align=max(1, dtype.width // 8),
    )


def _compile(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    direct_output: bool,
    return_lse: bool,
    kv_warps: int,
    selection_width: int = _SELECTION_WIDTH,
) -> tuple[tuple[object, ...], Callable[..., None]]:
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=direct_output,
        return_lse=return_lse,
        kv_warps=kv_warps,
        selection_width=selection_width,
    )
    with _LOCK:
        cached = _KERNEL_CACHE.get(key)
        if cached is not None:
            return key, cached
        (
            _,
            q_heads,
            kv_heads,
            request_id_bits,
            kv_dtype,
            page_size,
            key_strides,
            value_strides,
            direct_output,
            return_lse,
            kv_warps,
            selection_width,
        ) = key
        request_id_type = Int32 if request_id_bits == 32 else Int64
        kv_type = Float8E4M3FN if kv_dtype == torch.float8_e4m3fn else BFloat16
        kernel = _SelectedPositionPagedForwardKernel(
            q_heads=q_heads,
            kv_heads=kv_heads,
            kv_is_fp8=kv_dtype == torch.float8_e4m3fn,
            direct_output=direct_output,
            return_lse=return_lse,
            kv_warps=kv_warps,
            page_size=page_size,
            key_strides=key_strides,
            value_strides=value_strides,
            selection_width=selection_width,
        )
        with torch.cuda.device(key[0]):
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(BFloat16),
                _fake_pointer(kv_type),
                _fake_pointer(kv_type),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(Int32),
                _fake_pointer(request_id_type),
                _fake_pointer(Int32),
                _fake_pointer(Int64),
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                Int64(1),
                Int64(1),
                Int64(1),
                Float32(1.0),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.paged.selected_forward",
                    ((2104 if _TMA_V else 2004) if _TMA else (3004 if _SPLITQK else 1004) if _PIPELINE else 4) + (10000 if _SWZ333 else 0) + (20000 if _ROWMAP else 0),
                    (
                        q_heads,
                        kv_heads,
                        request_id_bits,
                        str(kv_dtype),
                        page_size,
                        key_strides,
                        value_strides,
                        direct_output,
                        return_lse,
                        kv_warps,
                        selection_width,
                    ),
                    labels=(
                        "q_heads",
                        "kv_heads",
                        "request_id_bits",
                        "kv_dtype",
                        "page_size",
                        "key_strides",
                        "value_strides",
                        "direct_output",
                        "return_lse",
                        "kv_warps",
                        "selection_width",
                    ),
                ),
            )
        _KERNEL_CACHE[key] = raw
        return key, raw


def precompile_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    request_ids: torch.Tensor,
    selection_width: int = _SELECTION_WIDTH,
) -> None:
    """Compile the caller's planned selection capacity without storage writes."""
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CuTe sparse GQA compilation is forbidden during capture"
            )
        _compile(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            request_ids=request_ids,
            direct_output=False,
            return_lse=True,
            kv_warps=2,
            selection_width=selection_width,
        )


def launch_sparse_gqa_split(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    softmax_scale: float,
    splits: int,
) -> None:
    """Launch the specialized split core without allocation or fallback."""
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=False,
        return_lse=True,
        kv_warps=2,
        selection_width=int(selected_positions.shape[1]),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA must be compiled and warm-run before CUDA graph "
                "capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=False,
                return_lse=True,
                kv_warps=2,
                selection_width=int(selected_positions.shape[1]),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _fake_pointer(BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_selected_paged_gqa_direct(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor | None,
    softmax_scale: float,
    kv_warps: int,
) -> None:
    """Launch direct selected-position paged attention without split scratch."""
    kv_warps = int(kv_warps)
    key = _cache_key(
        query,
        key_cache,
        value_cache,
        request_ids,
        direct_output=True,
        return_lse=output_lse is not None,
        kv_warps=kv_warps,
        selection_width=int(selected_positions.shape[1]),
    )
    with torch.cuda.device(query.device):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _KERNEL_CACHE.get(key)
            warmed = raw is not None and _WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "selected-position paged attention must be compiled and warm-run "
                "before CUDA graph capture"
            )
        if raw is None:
            key, raw = _compile(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                request_ids=request_ids,
                direct_output=True,
                return_lse=output_lse is not None,
                kv_warps=kv_warps,
                selection_width=int(selected_positions.shape[1]),
            )
        request_id_type = Int32 if request_ids.dtype == torch.int32 else Int64
        kv_type = Float8E4M3FN if key_cache.dtype == torch.float8_e4m3fn else BFloat16
        run_compiled(
            raw,
            (
                _pointer(query, BFloat16),
                _pointer(key_cache, kv_type),
                _pointer(value_cache, kv_type),
                _pointer(k_descale, Float32)
                if k_descale is not None
                else _fake_pointer(Float32),
                _pointer(v_descale, Float32)
                if v_descale is not None
                else _fake_pointer(Float32),
                _pointer(block_table, Int32),
                _pointer(request_ids, request_id_type),
                _pointer(selected_positions, Int32),
                _pointer(query_positions, Int64),
                _fake_pointer(Float32),
                _pointer(output_lse, Float32)
                if output_lse is not None
                else _fake_pointer(Float32),
                _pointer(output, BFloat16),
                int(key_cache.shape[0]),
                int(block_table.shape[0]),
                int(block_table.shape[1]),
                float(softmax_scale),
                int(query.shape[0]),
                1,
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _KERNEL_CACHE.get(key) is raw:
                _WARMED[key] = raw


def launch_sparse_gqa_merge(
    *,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    output: torch.Tensor,
    output_lse: torch.Tensor | None,
    rows: int,
    splits: int,
) -> None:
    """Merge the Qwen split layout without allocation or fallback."""
    device_index = partial_output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    q_heads = int(partial_output.shape[2])
    return_lse = output_lse is not None
    key = (
        int(device_index),
        q_heads,
        _HEAD_DIM,
        _MERGE_KERNEL.__name__,
        return_lse,
    )
    with torch.cuda.device(device_index):
        capturing = torch.cuda.is_current_stream_capturing()
        with _LOCK:
            raw = _MERGE_CACHE.get(key)
            warmed = raw is not None and _MERGE_WARMED.get(key) is raw
        if capturing and (raw is None or not warmed):
            raise RuntimeError(
                "CuTe sparse GQA merge must be compiled and warm-run before "
                "CUDA graph capture"
            )
        if raw is None:
            kernel = _MERGE_KERNEL(q_heads=q_heads, return_lse=return_lse)
            raise_if_kernel_resolution_frozen(
                "cute.compile",
                target=kernel,
                cache_key=key,
            )
            raw = b12x_compile(
                kernel,
                _fake_pointer(Float32),
                _fake_pointer(Float32),
                _fake_pointer(BFloat16),
                _fake_pointer(Float32),
                Int32(1),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.sparse_gqa_merge",
                    11,
                    key[1:],
                ),
            )
            with _LOCK:
                _MERGE_CACHE[key] = raw
        if compile_only_launches_enabled():
            for program in program_keys(raw):
                record_program(program)
            return raw
        run_compiled(
            raw,
            (
                _pointer(partial_output, Float32),
                _pointer(partial_lse, Float32),
                _pointer(output, BFloat16),
                _pointer(output_lse, Float32)
                if output_lse is not None
                else _fake_pointer(Float32),
                int(rows),
                int(splits),
                current_cuda_stream(),
            ),
        )
    if not capturing:
        with _LOCK:
            if _MERGE_CACHE.get(key) is raw:
                _MERGE_WARMED[key] = raw


def clear_caches() -> None:
    """Clear process-local compiled and warm-launch state for focused tests."""
    with _LOCK:
        _KERNEL_CACHE.clear()
        _WARMED.clear()
        _MERGE_CACHE.clear()
        _MERGE_WARMED.clear()


__all__ = [
    "clear_caches",
    "launch_selected_paged_gqa_direct",
    "launch_sparse_gqa_split",
    "launch_sparse_gqa_merge",
    "precompile_sparse_gqa_split",
]

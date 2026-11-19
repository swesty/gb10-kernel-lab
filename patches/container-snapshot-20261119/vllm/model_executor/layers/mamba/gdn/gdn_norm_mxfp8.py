"""Gated RMSNorm (group mode) with an MXFP8 epilogue, for the GDN out_proj input.

The norm part is a copy of vLLM's FLA ``layer_norm_fwd_kernel`` (RMS, gate after the norm), launched with the
same tile (4 rows x 128 columns per program, one warp), so its reductions and therefore its BF16 output are the
same. Instead of storing BF16, it rounds y to BF16 and quantises it exactly like b12x's Triton ``_quantize``
(MXFP8: one E8M0 scale per 32 values, scale = 2^ceil(log2(amax / 448)), values = y / scale as e4m3), writing
the values and the swizzled scale bytes straight into the GEMM's workspace. Rows in [live, padded) are treated
as zero rows, exactly as the zero-filled tail of ``core_attn_out`` would be.
Installed into the container as vllm/model_executor/layers/mamba/gdn/gdn_norm_mxfp8.py.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _scale_offset(row, group, GROUPS: tl.constexpr):
    return (
        (row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + group % 4
    )


@triton.jit
def _gated_rmsnorm_mxfp8_kernel(
    X, Z, W, Q, S, ALPHA,
    stride_x_row, stride_z_row,
    LIVE, M,  # live rows (real data) and rows of the GEMM input
    eps,
    N: tl.constexpr,  # group size (head dim)
    K: tl.constexpr,  # row width = groups * N
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)
    col_offsets = cols[None, :] + group * N
    row_mask = rows[:, None] < LIVE
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask
    # ---- identical to layer_norm_fwd_kernel (IS_RMS_NORM, NORM_BEFORE_GATE, HAS_Z, no bias) ----
    x = tl.load(X + rows[:, None] * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols + group * N, mask=cols < N, other=0.0).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(Z + rows[:, None] * stride_z_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    # ---- BF16 rounding as the stored output would have, then b12x _quantize (MXFP8) ----
    y = tl.where(mask, y, 0.0).to(tl.bfloat16).to(tl.float32)
    # Per-32 amax without reshaping y, so the tile layout (and the norm's reduction order) stays the same.
    rows64 = rows.to(tl.int64)
    ay = tl.abs(y)
    qcol = cols // 32
    scale_cols = tl.zeros((ROWS_PER_BLOCK, BLOCK_N), dtype=tl.float32)
    for g in tl.static_range(BLOCK_N // 32):
        amax = tl.max(tl.where(qcol[None, :] == g, ay, 0.0), axis=1)
        ratio = tl.where(amax > 0, amax / 448.0, 1.0)
        exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
        scale_cols = tl.where(qcol[None, :] == g, tl.exp2(exponent)[:, None], scale_cols)
        sf_byte = (exponent + 127).to(tl.uint8)
        tl.store(S + _scale_offset(rows64, group * (BLOCK_N // 32) + g, K // 32), sf_byte)
    values = (y / scale_cols).to(tl.float8e4nv)
    tl.store(Q + rows64[:, None] * K + col_offsets, values, mask=(rows[:, None] < M) & col_mask)
    if (tl.program_id(0) == 0) & (group == 0):
        tl.store(ALPHA, 1.0)


def gated_rmsnorm_mxfp8(x, z, weight, eps, live, m, values, scales, alpha, *, group_size, activation):
    """x: [live, K] BF16 (row stride free), z: [live, K] view with unit column stride, weight: [K] tiled.

    values: FP8 [>= m*K] workspace view; scales: uint8 swizzled region for ceil(m/128)*128 rows; alpha: FP32 [1].
    """
    k = weight.numel()
    assert k % group_size == 0 and group_size % 32 == 0 and group_size <= 1024
    padded = triton.cdiv(m, 128) * 128
    # same as FLA calc_rows_per_block(live): tile shape fixes the reduction order
    rows_per_block = min(triton.next_power_of_2(triton.cdiv(max(live, 1), 2 * torch.cuda.get_device_properties(x.device).multi_processor_count)), 4)
    grid = (triton.cdiv(padded, rows_per_block), k // group_size)
    _gated_rmsnorm_mxfp8_kernel[grid](
        x, z, weight, values, scales, alpha,
        x.stride(0), z.stride(0), live, m, eps,
        N=group_size, K=k, BLOCK_N=triton.next_power_of_2(group_size), ROWS_PER_BLOCK=rows_per_block,
        ACTIVATION=activation, num_warps=1,
    )

"""Why does dense_gemm on harness-built MXFP8 operands run slower than the live path? Compare A-scale layouts."""
import sys, torch
sys.path.insert(0, "/tmp/gemmexp")
from dense_gemm_harness import Problem8, kernel_times, DEV
from b12x._lib.dense_gemm import dense_gemm
from b12x._lib.intrinsics import as_grouped_scale_view_mx
from b12x.gemm.blockscaled._linear import _mxfp8_scale_mma_from_input
from b12x.gemm.blockscaled._a16 import _layout
import triton

name = sys.argv[1] if len(sys.argv) > 1 else "out"
p = Problem8(name, 4096)
m, n, k = p.m, p.n, p.k
ws = p.workspace()
live_out = p.api.mm(p.x, p.weight, workspace=ws); torch.cuda.synchronize()
sf_start, alpha_start, partial_start, needed = _layout(m, n, k, False, (64, 64, 1))
size = triton.cdiv(m, 128) * triton.cdiv(k // 32, 4) * 512
a_ws = ws[: m * k].view(torch.float8_e4m3fn).view(m, k, 1)
s_ws = as_grouped_scale_view_mx(ws[sf_start: sf_start + size].view(1, -1), m, k)
xq, xs = p.quant()
s_mine = _mxfp8_scale_mma_from_input(xs, rows=m, width=k, logical_width=k).view(torch.float8_e8m0fnu)
print("ws scale view", tuple(s_ws.shape), s_ws.stride(), s_ws.dtype, "data_ptr%16", s_ws.data_ptr() % 16)
print("my scale view", tuple(s_mine.shape), s_mine.stride(), s_mine.dtype, "data_ptr%16", s_mine.data_ptr() % 16)
print("A values equal:", torch.equal(a_ws.view(torch.uint8).view(m, k), xq.view(torch.uint8)))
w = p.weight.weight
b3, b_mma = w.values.view(n, k, 1), w.scale_mma.view(torch.float8_e8m0fnu)
print("B scale", tuple(b_mma.shape), b_mma.stride())
alpha = torch.ones(1, device=DEV, dtype=torch.float32)
out = torch.empty((m, n, 1), device=DEV, dtype=torch.bfloat16)
for label, a, s in (("ws operands", a_ws, s_ws), ("my operands", xq.view(m, k, 1), s_mine)):
    fn = lambda: dense_gemm((a, s), (b3, b_mma), out, ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu",
                            c_dtype="bfloat16", sf_vec_size=32, alpha=alpha, expected_m=4096)
    fn(); torch.cuda.synchronize()
    kt = kernel_times(fn, 8, cold=True)
    print(label, {kk[-60:]: round(v[0], 1) for kk, v in kt.items()}, "err vs live",
          ((out[:, :, 0].float() - live_out.float()).norm() / live_out.float().norm()).item())
kt = kernel_times(lambda: p.api.mm(p.x, p.weight, workspace=ws), 8, cold=True)
print("live", {kk[-60:]: round(v[0], 1) for kk, v in kt.items()})

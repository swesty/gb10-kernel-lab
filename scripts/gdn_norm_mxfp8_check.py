"""Byte-level check: fused gated RMSNorm + MXFP8 vs FLA layer_norm_fwd -> b12x _quantize (GDN out_proj input)."""
import sys, torch, triton
sys.path.insert(0, "/tmp")
from gdn_norm_mxfp8 import gated_rmsnorm_mxfp8
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
from b12x.gemm.blockscaled._quantize import _quantize

dev = torch.device("cuda", 0)
H, D = 48, 128
K = H * D
torch.manual_seed(0)
w128 = (torch.randn(D, device=dev) * 0.3 + 1.0).to(torch.bfloat16)
weight = w128.repeat(H)
for live, m in ((4096, 4096), (3000, 3072), (77, 128), (4096, 4100)):
    x = (torch.randn(live, K, device=dev) * 2.0).to(torch.bfloat16)
    zbase = (torch.randn(live, 2 * K, device=dev) * 3.0).to(torch.bfloat16)
    z = zbase.as_strided((live, K), (2 * K, 1))          # strided gate like the live path
    # reference: BF16 norm output (zero tail), then b12x _quantize
    out = torch.zeros(m, K, device=dev, dtype=torch.bfloat16)
    layer_norm_fwd(x, weight, None, 1e-6, z=z, out=out[:live], group_size=D, norm_before_gate=True,
                   is_rms_norm=True, activation="silu")
    groups = K // 32
    ssize = triton.cdiv(m, 128) * triton.cdiv(groups, 4) * 512
    q_ref = torch.full((m * K,), 0x7F, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    s_ref = torch.full((ssize,), 0xAB, dtype=torch.uint8, device=dev)
    a_ref = torch.zeros(1, device=dev)
    _quantize[(m, triton.cdiv(groups, 16), 1)](out, q_ref, s_ref, a_ref, a_ref, a_ref, m,
                                               INPUT_K=K, K=K, FP4=False, RECIPROCAL=False, GROUP=32, CHUNKS=16,
                                               num_warps=4, num_stages=1, enable_fp_fusion=False)
    q = torch.full((m * K,), 0x7F, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
    s = torch.full((ssize,), 0xCD, dtype=torch.uint8, device=dev)
    a = torch.zeros(1, device=dev)
    gated_rmsnorm_mxfp8(x, z, weight, 1e-6, live, m, q, s, a, group_size=D, activation="silu")
    torch.cuda.synchronize()
    qv = (q.view(torch.uint8) == q_ref.view(torch.uint8)).float().mean().item()
    sv = (s == s_ref).float().mean().item()
    print(f"live={live} m={m}: values equal {100*qv:.5f}%  scales equal {100*sv:.5f}%  alpha {a.item()} vs {a_ref.item()}")
    # timing at the live shape
    if live == 4096 and m == 4096:
        def t(fn, n=20):
            fn(); torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(n): fn()
            e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / n * 1000
        ref_norm = t(lambda: layer_norm_fwd(x, weight, None, 1e-6, z=z, out=out[:live], group_size=D, norm_before_gate=True, is_rms_norm=True, activation="silu"))
        ref_q = t(lambda: _quantize[(m, triton.cdiv(groups, 16), 1)](out, q_ref, s_ref, a_ref, a_ref, a_ref, m, INPUT_K=K, K=K, FP4=False, RECIPROCAL=False, GROUP=32, CHUNKS=16, num_warps=4, num_stages=1, enable_fp_fusion=False))
        fused = t(lambda: gated_rmsnorm_mxfp8(x, z, weight, 1e-6, live, m, q, s, a, group_size=D, activation="silu"))
        print(f"  norm {ref_norm:.0f} us + quantize {ref_q:.0f} us = {ref_norm+ref_q:.0f} us  vs fused {fused:.0f} us")

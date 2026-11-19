"""Byte-level check: fused attention gate + MXFP8 vs _qsa_apply_gate -> b12x _quantize (o_proj input)."""
import torch, triton
import vllm.models.qwen4_exp.nvidia.b12x_qsa as qs
from b12x.gemm.blockscaled._quantize import _quantize

dev = torch.device("cuda", 0)
H, D = 24, 256
K = H * D
torch.manual_seed(0)
for m in (4096, 2282, 100):
    out = (torch.randn(m, H, D, device=dev) * 1.5).to(torch.bfloat16)
    q_gate = (torch.randn(m, H * 2 * D, device=dev) * 2.0).to(torch.bfloat16)
    gate = qs._PackedGate(q_gate, D)
    ref_bf16 = qs._qsa_apply_gate(out, gate)
    G = K // 32
    ss = triton.cdiv(m, 128) * triton.cdiv(G, 4) * 512
    qr = torch.empty(m * K, dtype=torch.float8_e4m3fn, device=dev); sr = torch.full((ss,), 0xAB, dtype=torch.uint8, device=dev)
    a = torch.zeros(1, device=dev)
    _quantize[(m, triton.cdiv(G, 16), 1)](ref_bf16, qr, sr, a, a, a, m, INPUT_K=K, K=K, FP4=False, RECIPROCAL=False,
                                          GROUP=32, CHUNKS=16, num_warps=4, num_stages=1, enable_fp_fusion=False)
    q = torch.empty_like(qr); s = torch.full((ss,), 0xCD, dtype=torch.uint8, device=dev)
    prod = qs._QsaGateQuantProducer(out.flatten(-2), gate)
    prod(q, s, a, m)
    torch.cuda.synchronize()
    print(f"m={m}: values equal {100 * (q.view(torch.uint8) == qr.view(torch.uint8)).float().mean().item():.5f}%  "
          f"scales equal {100 * (s == sr).float().mean().item():.5f}%")

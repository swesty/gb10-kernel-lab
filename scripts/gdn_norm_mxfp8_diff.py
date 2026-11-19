"""Show the elements where fused and reference MXFP8 bytes differ."""
import sys, torch, triton
sys.path.insert(0, "/tmp")
from gdn_norm_mxfp8 import gated_rmsnorm_mxfp8
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
from b12x.gemm.blockscaled._quantize import _quantize
dev = torch.device("cuda", 0); H, D = 48, 128; K = H * D; live = m = 4096
torch.manual_seed(0)
weight = (torch.randn(D, device=dev) * 0.3 + 1.0).to(torch.bfloat16).repeat(H)
x = (torch.randn(live, K, device=dev) * 2.0).to(torch.bfloat16)
z = (torch.randn(live, 2 * K, device=dev) * 3.0).to(torch.bfloat16).as_strided((live, K), (2 * K, 1))
out = torch.zeros(m, K, device=dev, dtype=torch.bfloat16)
layer_norm_fwd(x, weight, None, 1e-6, z=z, out=out, group_size=D, norm_before_gate=True, is_rms_norm=True, activation="silu")
G = K // 32; ss = triton.cdiv(m, 128) * triton.cdiv(G, 4) * 512
qr = torch.empty(m * K, dtype=torch.float8_e4m3fn, device=dev); sr = torch.empty(ss, dtype=torch.uint8, device=dev); a = torch.zeros(1, device=dev)
_quantize[(m, triton.cdiv(G, 16), 1)](out, qr, sr, a, a, a, m, INPUT_K=K, K=K, FP4=False, RECIPROCAL=False, GROUP=32, CHUNKS=16, num_warps=4, num_stages=1, enable_fp_fusion=False)
q = torch.empty_like(qr); s = torch.empty_like(sr)
gated_rmsnorm_mxfp8(x, z, weight, 1e-6, live, m, q, s, a, group_size=D, activation="silu")
torch.cuda.synchronize()
d = (q.view(torch.uint8) != qr.view(torch.uint8)).nonzero().flatten()
for i in d[:10].tolist():
    r, c = divmod(i, K)
    # rebuild the scale for that group from the reference quantiser: scale = 2^(byte-127)
    g = c // 32
    off = ((r // 128) * triton.cdiv(G, 4) + g // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + g % 4
    sc = 2.0 ** (int(sr[off]) - 127)
    print(f"row {r} col {c}: ref bf16 {out[r, c].item():+.6f} -> y/scale {out[r, c].float().item() / sc:+.6f}; "
          f"ref fp8 {qr[i].float().item():+.4f} fused fp8 {q[i].float().item():+.4f}")
print(len(d), "differing bytes")

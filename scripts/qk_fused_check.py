"""Eager (GemmaRMSNorm + mRoPE, as b12x_qsa) vs vLLM fused_qk_rmsnorm_rope_gate at Flash Next shapes."""
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
torch.manual_seed(0)
H, KV, D = 24, 2, 256
with set_current_vllm_config(VllmConfig()):
    rope = get_rope(head_size=D, max_position=262144, rope_parameters={
        "mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
        "rope_theta": 10000000, "rope_type": "default"}, dtype=torch.bfloat16).cuda()
    qn = GemmaRMSNorm(D, eps=1e-6).cuda(); kn = GemmaRMSNorm(D, eps=1e-6).cuda()
qn.weight.data.normal_(0, 0.1); kn.weight.data.normal_(0, 0.1)
print(type(rope).__name__, "rotary_dim", rope.rotary_dim, "neox", rope.is_neox_style, "dtype", rope.dtype)
for T in (5, 4096):
    qkv = (torch.randn(T, H * 2 * D + 2 * KV * D, device="cuda") * 2).to(torch.bfloat16)
    p = torch.arange(10000, 10000 + T, device="cuda")
    pos = torch.stack([p, p, p])  # text-only mRoPE positions
    q_gate, key, value = qkv.split((H * 2 * D, KV * D, KV * D), dim=-1)
    qg = q_gate.unflatten(-1, (H, 2 * D)); q, g = qg.chunk(2, dim=-1)
    q_e = qn(q).flatten(-2); k_e = kn(key.unflatten(-1, (KV, D))).flatten(-2)
    q_e, k_e = rope(pos, q_e, k_e)
    q_f, k_f, g_f = fused_qk_rmsnorm_rope_gate(q_gate, key, qn.weight, kn.weight, rope.cos_sin_cache, pos,
                                               qn.variance_epsilon, H, KV, D, rope.rotary_dim, norm_beta=1.0,
                                               mrope_section=rope.mrope_section)
    for name, a, b in (("q", q_e, q_f), ("k", k_e, k_f), ("gate", g.flatten(-2), g_f)):
        a = a.float(); b = b.float(); d = (a - b).abs()
        print(f"T={T} {name}: mismatched {(d > 0).float().mean().item()*100:.4f}%  max {d.max().item():.4f}  "
              f"(norm-only cols differ {(d[:, :].reshape(T, -1, D)[..., 64:] > 0).float().mean().item()*100:.4f}%)")
    torch.cuda.synchronize()
    import time
    def eager():
        qg = q_gate.unflatten(-1, (H, 2 * D)); q, g = qg.chunk(2, dim=-1)
        a = qn(q).flatten(-2); b = kn(key.unflatten(-1, (KV, D))).flatten(-2); return rope(pos, a, b)
    def fused():
        return fused_qk_rmsnorm_rope_gate(q_gate, key, qn.weight, kn.weight, rope.cos_sin_cache, pos, 1e-6, H, KV, D,
                                          rope.rotary_dim, norm_beta=1.0, mrope_section=rope.mrope_section)
    for nm, f in (("eager", eager), ("fused", fused)):
        for _ in range(3): f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(20): f()
        torch.cuda.synchronize(); print(f"  {nm}: {(time.time()-t)/20*1e6:.0f} us")

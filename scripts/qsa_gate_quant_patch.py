"""Fuse the attention output gate with o_proj's MXFP8 activation quantisation (switch VLLM_QSA_FUSED_GATE_QUANT=1).

Same mechanism as the GDN norm fusion (norm_quant_fusion_patch.py, which adds b12x's B12X_MXFP8_PREQUANT hook):
o_proj runs with a producer that computes bf16(o * bf16(sigmoid(g))) from the attention output and the gate
(read in place from the packed [q | gate] projection) and writes the MXFP8 values and swizzled scales straight
into the GEMM workspace. Row counts that ever fall back are not deferred again; small batches (<= 80 rows,
the CUDA-graph sizes, which b12x serves with exact-size states) are never deferred.
Usage (in the container, after norm_quant_fusion_patch.py): python3 qsa_gate_quant_patch.py
"""
import os
import shutil

import vllm.models.qwen4_exp.nvidia.b12x_qsa as qs

MARK = "QSA_GATE_QUANT_PATCH"
p = qs.__file__
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p).read()
if MARK in s:
    print("already patched", p)
    raise SystemExit

kernel = '''

# QSA_GATE_QUANT_PATCH: output gate + MXFP8 quantisation for o_proj, written into b12x's GEMM workspace.
_QSA_FUSED_GATE_QUANT = __import__("os").environ.get("VLLM_QSA_FUSED_GATE_QUANT", "0") == "1"
_QSA_GQ_NO_DEFER: set = set()


@triton.jit
def _qsa_gq_scale_offset(row, group, GROUPS: tl.constexpr):
    return ((row // 128 * triton.cdiv(GROUPS, 4) + group // 4) * 512
            + row % 32 * 16 + row // 32 % 4 * 4 + group % 4)


@triton.jit
def _qsa_gate_mxfp8_kernel(out_ptr, qg_ptr, Q, S, ALPHA, out_stride, qg_stride, M,
                           HEAD_DIM: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < WIDTH
    valid = mask & (row < M)
    head = cols // HEAD_DIM
    gcols = head * (2 * HEAD_DIM) + HEAD_DIM + cols % HEAD_DIM
    o = tl.load(out_ptr + row * out_stride + cols, mask=valid, other=0.0).to(tl.float32)
    g = tl.load(qg_ptr + row * qg_stride + gcols, mask=valid, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
    v = tl.where(valid, (o * sig).to(tl.bfloat16).to(tl.float32), 0.0)
    v2 = tl.reshape(v, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(v2), 1)
    ratio = tl.where(amax > 0, amax / 448.0, 1.0)
    exponent = tl.minimum(tl.maximum(tl.ceil(tl.log2(ratio)), -127.0), 127.0)
    vals = tl.reshape((v2 / tl.exp2(exponent)[:, None]).to(tl.float8e4nv), (BLOCK,))
    tl.store(Q + row * WIDTH + cols, vals, mask=valid)
    groups = tl.program_id(1) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(S + _qsa_gq_scale_offset(row, groups, WIDTH // 32), (exponent + 127).to(tl.uint8),
             mask=groups < WIDTH // 32)
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tl.store(ALPHA, 1.0)


class _QsaGateQuantProducer:
    def __init__(self, flat, gate):
        self.flat, self.gate = flat, gate
        self.m, self.k = flat.shape
        self.consumed = False

    def __call__(self, values, scales, alpha, m):
        padded = triton.cdiv(m, 128) * 128
        _qsa_gate_mxfp8_kernel[(padded, triton.cdiv(self.k, 1024))](
            self.flat, self.gate.q_gate, values, scales, alpha, self.flat.stride(0),
            self.gate.q_gate.stride(0), m, HEAD_DIM=self.gate.head_dim, WIDTH=self.k, BLOCK=1024,
            num_warps=4,
        )
        self.consumed = True


def _qsa_gated_o_proj(layer, output, gate):
    flat = output.flatten(-2)
    rows, width = flat.shape
    if (_QSA_FUSED_GATE_QUANT and isinstance(gate, _PackedGate) and rows > 80
            and rows not in _QSA_GQ_NO_DEFER and width % 1024 == 0 and flat.stride(-1) == 1
            and flat.dtype == torch.bfloat16 and not torch.cuda.is_current_stream_capturing()):
        from b12x.gemm.blockscaled._preparation import B12X_MXFP8_PREQUANT
        producer = _QsaGateQuantProducer(flat, gate)
        dummy = torch.empty((rows, width), dtype=flat.dtype, device=flat.device)
        token = B12X_MXFP8_PREQUANT.set(producer)
        try:
            projected = layer.o_proj(dummy)[0]
        finally:
            B12X_MXFP8_PREQUANT.reset(token)
        if producer.consumed:
            return projected
        _QSA_GQ_NO_DEFER.add(rows)
        print(f"[qsa gate-quant fusion] rows={rows} fell back; not deferring this size again", flush=True)
    return layer.o_proj(_qsa_apply_gate(output, gate))[0]
'''

reps = [
    ("\n\nclass _PackedGate:\n", kernel + "\n\nclass _PackedGate:\n"),
    ("            gated = _qsa_apply_gate(output, gate)\n            return self.o_proj(gated)[0]\n",
     "            return _qsa_gated_o_proj(self, output, gate)\n"),
    ("        gated = _qsa_apply_gate(output, gate)\n        projected, _ = self.o_proj(gated)\n        return projected\n",
     "        return _qsa_gated_o_proj(self, output, gate)\n"),
]
for old, new in reps:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)

"""Fuse the GDN gated RMSNorm with out_proj's MXFP8 activation quantisation (switch VLLM_GDN_FUSED_NORM_QUANT=1).

1. b12x (gemm/blockscaled/_preparation.py): a ContextVar ``B12X_MXFP8_PREQUANT``. When set for a call whose
   M and K match, the packed MXFP8 execution state calls it to fill its workspace (values, swizzled scales,
   alpha) instead of launching its own ``_quantize`` kernel. Nothing else changes.
2. vLLM (qwen_gdn_linear_attn.py): in the eager identity fast path (§13.6), if the caller allowed it, the
   gated norm is deferred: the job is kept on the layer and ``core_attn_out`` is left unwritten. The forward
   then runs ``out_proj`` with the ContextVar set to the fused kernel (gdn_norm_mxfp8.py), which writes the
   MXFP8 input straight into the GEMM workspace. If the hook was not used, the norm is materialised into
   ``core_attn_out`` and ``out_proj`` runs again, so the result is always produced.
Usage (in the container): python3 norm_quant_fusion_patch.py   (copies gdn_norm_mxfp8.py from /tmp)
"""
import os
import shutil

import b12x.gemm.blockscaled._preparation as bp
import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gd

MARK = "NORM_QUANT_FUSION_PATCH"


def patch(mod, reps):
    p = mod.__file__
    if not os.path.exists(p + ".orig"):
        shutil.copy(p, p + ".orig")
    s = open(p).read()
    if MARK in s:
        print("already patched", p)
        return
    for old, new in reps:
        assert s.count(old) == 1, (p, old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


shutil.copy("/tmp/gdn_norm_mxfp8.py", os.path.join(os.path.dirname(gd.__file__), "gdn_norm_mxfp8.py"))

patch(bp, [
    ("@dataclass(frozen=True)\nclass _PackedExecutionState:\n",
     "import contextvars as _cv_pq\n"
     "# NORM_QUANT_FUSION_PATCH: a caller may produce the MXFP8 activations itself (see run()).\n"
     "B12X_MXFP8_PREQUANT = _cv_pq.ContextVar(\"b12x_mxfp8_prequant\", default=None)\n\n\n"
     "@dataclass(frozen=True)\nclass _PackedExecutionState:\n"),
    ("            self.programs[\"quantize\"][grid](\n"
     "                source, packed_values, packed_scales, activation_scale if fp4 else alpha,\n"
     "                global_scale if fp4 else alpha, alpha, m, *constants,\n"
     "            )\n",
     "            _pq = None if fp4 else B12X_MXFP8_PREQUANT.get()\n"
     "            if _pq is not None and _pq.m == m and _pq.k == q.padded_in_features == q.in_features:\n"
     "                _pq(packed_values, packed_scales, alpha, m)\n"
     "            else:\n"
     "                self.programs[\"quantize\"][grid](\n"
     "                    source, packed_values, packed_scales, activation_scale if fp4 else alpha,\n"
     "                    global_scale if fp4 else alpha, alpha, m, *constants,\n"
     "                )\n"),
])

patch(gd, [
    # identity path: defer the norm when the caller allows it
    ("        gate = output_gate[:live]\n"
     "        gate_2d = gate.as_strided((live, heads * dim), (gate.stride(0), 1))\n"
     "        assert gate.stride(-1) == 1 and gate.stride(-2) == dim\n"
     "        layer_norm_fwd(\n",
     "        gate = output_gate[:live]\n"
     "        gate_2d = gate.as_strided((live, heads * dim), (gate.stride(0), 1))\n"
     "        assert gate.stride(-1) == 1 and gate.stride(-2) == dim\n"
     "        # NORM_QUANT_FUSION_PATCH: hand the norm to out_proj's MXFP8 quantisation when allowed.\n"
     "        if (getattr(self, \"_gdn_defer_norm_ok\", False) and bias is None and self.norm.norm_before_gate\n"
     "                and group % 32 == 0 and self.norm.activation in (\"silu\", \"swish\", \"sigmoid\")):\n"
     "            self._gdn_pending_norm = dict(\n"
     "                x=prefill_output[:live].reshape(live, heads * dim), z=gate_2d, weight=weight,\n"
     "                eps=self.norm.eps, live=live, rows=rows, group=group, activation=self.norm.activation,\n"
     "                out=core_attn_out)\n"
     "            return\n"
     "        layer_norm_fwd(\n"),
    # forward: run out_proj with the fused producer
    ("            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(\n"
     "                mixed_qkvz,\n"
     "                ba,\n"
     "                core_attn_out,\n"
     "                layer_name=_encode_layer_name(self.prefix),\n"
     "            )\n"
     "            output, _ = self.out_proj(core_attn_out.flatten(-2))\n"
     "            return output\n",
     "            self._gdn_defer_norm_ok = _GDN_FUSED_NORM_QUANT and not torch.cuda.is_current_stream_capturing()\n"
     "            try:\n"
     "                torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(\n"
     "                    mixed_qkvz,\n"
     "                    ba,\n"
     "                    core_attn_out,\n"
     "                    layer_name=_encode_layer_name(self.prefix),\n"
     "                )\n"
     "            finally:\n"
     "                self._gdn_defer_norm_ok = False\n"
     "            job = self.__dict__.pop(\"_gdn_pending_norm\", None)\n"
     "            if job is None:\n"
     "                output, _ = self.out_proj(core_attn_out.flatten(-2))\n"
     "                return output\n"
     "            return _gdn_out_proj_fused(self, job, core_attn_out)\n"),
    ("@PluggableLayer.register(\"qwen_gated_delta_net_attention\")\nclass QwenGatedDeltaNetAttention(",
     "import os as _os_nq\n"
     "_GDN_FUSED_NORM_QUANT = _os_nq.environ.get(\"VLLM_GDN_FUSED_NORM_QUANT\", \"0\") == \"1\"\n\n\n"
     "class _GdnNormQuantProducer:\n"
     "    \"\"\"NORM_QUANT_FUSION_PATCH: fills b12x's MXFP8 workspace from the deferred gated norm.\"\"\"\n\n"
     "    def __init__(self, job):\n"
     "        self.job, self.m, self.k, self.consumed = job, job[\"rows\"], job[\"weight\"].numel(), False\n\n"
     "    def __call__(self, values, scales, alpha, m):\n"
     "        from .gdn_norm_mxfp8 import gated_rmsnorm_mxfp8\n"
     "        j = self.job\n"
     "        gated_rmsnorm_mxfp8(j[\"x\"], j[\"z\"], j[\"weight\"], j[\"eps\"], j[\"live\"], m, values, scales, alpha,\n"
     "                            group_size=j[\"group\"], activation=j[\"activation\"])\n"
     "        self.consumed = True\n\n\n"
     "def _gdn_materialize_norm(job):\n"
     "    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd\n"
     "    out, live, rows = job[\"out\"], job[\"live\"], job[\"rows\"]\n"
     "    layer_norm_fwd(job[\"x\"], job[\"weight\"], None, job[\"eps\"], z=job[\"z\"],\n"
     "                   out=out[:live].view(live, -1), group_size=job[\"group\"], norm_before_gate=True,\n"
     "                   is_rms_norm=True, activation=job[\"activation\"])\n"
     "    if live < rows:\n"
     "        out[live:].zero_()\n\n\n"
     "def _gdn_out_proj_fused(layer, job, core_attn_out):\n"
     "    from b12x.gemm.blockscaled._preparation import B12X_MXFP8_PREQUANT\n"
     "    producer = _GdnNormQuantProducer(job)\n"
     "    token = B12X_MXFP8_PREQUANT.set(producer)\n"
     "    try:\n"
     "        output, _ = layer.out_proj(core_attn_out.flatten(-2))\n"
     "    finally:\n"
     "        B12X_MXFP8_PREQUANT.reset(token)\n"
     "    if not producer.consumed:\n"
     "        _gdn_materialize_norm(job)\n"
     "        output, _ = layer.out_proj(core_attn_out.flatten(-2))\n"
     "        if not getattr(layer, \"_gdn_nq_warned\", False):\n"
     "            layer._gdn_nq_warned = True\n"
     "            print(\"[gdn norm-quant fusion] hook not used; fell back to norm + out_proj\", flush=True)\n"
     "    else:\n"
     "        _GDN_NQ_STATS[\"fused\"] += 1\n"
     "    return output\n\n\n"
     "_GDN_NQ_STATS = {\"fused\": 0}\n\n\n"
     "@PluggableLayer.register(\"qwen_gated_delta_net_attention\")\nclass QwenGatedDeltaNetAttention("),
])

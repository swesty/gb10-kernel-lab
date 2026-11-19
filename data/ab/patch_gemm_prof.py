"""Add an env-gated per-call GEMM timer to vLLM's b12x NVFP4 linear (VLLM_B12X_GEMM_PROF=1).

Trigger files in /root/.cache/vllm/nsys-flashnext/: GEMMPROF_RESET clears the record, GEMMPROF_DUMP writes
gemmprof.json (per layer class, path, M, N, K: calls and total ms) and clears. Only eager calls with
M >= 512 and no stream capture are timed. Run inside the container: python3 patch_gemm_prof.py <file>
"""
import sys

p = sys.argv[1]
s = open(p).read()
assert "_GEMM_PROF" not in s, "already patched"

helper = '''
import json as _json
import os as _os
import re as _re

_GEMM_PROF = _os.environ.get("VLLM_B12X_GEMM_PROF") == "1"
_GP_DIR = "/root/.cache/vllm/nsys-flashnext"
_gp_events: list = []


def _gp_check_triggers() -> None:
    if _os.path.exists(_GP_DIR + "/GEMMPROF_RESET"):
        _gp_events.clear()
        _os.remove(_GP_DIR + "/GEMMPROF_RESET")
    if _os.path.exists(_GP_DIR + "/GEMMPROF_DUMP"):
        torch.cuda.synchronize()
        agg: dict = {}
        for name, path, m, n, k, s, e in _gp_events:
            key = f"{name}|{path}|{m}|{n}|{k}"
            a = agg.setdefault(key, [0, 0.0])
            a[0] += 1
            a[1] += s.elapsed_time(e)
        with open(_GP_DIR + "/gemmprof.json", "w") as f:
            _json.dump(agg, f, indent=1)
        _gp_events.clear()
        _os.remove(_GP_DIR + "/GEMMPROF_DUMP")


def _gp_time(layer, path, m, n, k, fn):
    if not _GEMM_PROF:
        return fn()
    _gp_check_triggers()
    if m < 512 or torch.cuda.is_current_stream_capturing():
        return fn()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    name = _re.sub(r"\\.\\d+\\.", ".#.", str(getattr(layer, "b12x_layer_name", "?")))
    _gp_events.append((name, path, m, n, k, s, e))
    return out

'''

anchor = "def _apply_b12x_nvfp4_linear("
assert s.count(anchor) == 1
s = s.replace(anchor, helper + "\n" + anchor)

old_bf16 = '''        output = run_b12x_blockscaled_linear(
            source,
            bias,
            out_features,
            layer.b12x_layer_name,
        )
        return output.view(*output_shape)'''
new_bf16 = '''        output = _gp_time(layer, "bf16-fused", x_2d.shape[0], out_features, x_2d.shape[1],
                          lambda: run_b12x_blockscaled_linear(
            source,
            bias,
            out_features,
            layer.b12x_layer_name,
        ))
        return output.view(*output_shape)'''
assert s.count(old_bf16) == 1
s = s.replace(old_bf16, new_bf16)

old_q = '''    x_packed, x_scale_swizzled = scaled_fp4_quant(
        x_2d,
        layer.input_global_scale_inv,
        is_sf_swizzled_layout=True,
    )
    output = run_b12x_nvfp4_serialized_linear(
        x_packed,
        x_scale_swizzled,
        bias,
        output_size,
        x.dtype,
        layer.b12x_layer_name,
    )'''
new_q = '''    x_packed, x_scale_swizzled = _gp_time(layer, "a4-quant", x_2d.shape[0], output_size, x_2d.shape[1],
                                          lambda: scaled_fp4_quant(
        x_2d,
        layer.input_global_scale_inv,
        is_sf_swizzled_layout=True,
    ))
    output = _gp_time(layer, "a4-mm", x_2d.shape[0], output_size, x_2d.shape[1],
                      lambda: run_b12x_nvfp4_serialized_linear(
        x_packed,
        x_scale_swizzled,
        bias,
        output_size,
        x.dtype,
        layer.b12x_layer_name,
    ))'''
assert s.count(old_q) == 1
s = s.replace(old_q, new_q)
open(p, "w").write(s)
print("patched", p)

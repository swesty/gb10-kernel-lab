"""Per-shape tile override for b12x's quantized-mode dense GEMM (MXFP8 prefill).

B12X_DENSE_TILE_OVERRIDE="<N>x<K>=<tm>x<tn>x<tk>[:lmu],..." (default unset = stock). When b12x lowers a
quantized-mode BlockscaledQuery with out_features N and padded in_features K for a large-M (prefill) plan
(expected_m >= 2048), the listed tile is passed to its DenseGemmQuery as launch overrides (mma_tiler_mn,
_tile_k_override, and _large_m_unroll_override when ":lmu" is given). The overrides change the lowering, so
they reach every compile/preparation key. b12x's own tuner cannot choose these tiles in this mode.
Usage (in the container): python3 dense_tile_patch.py   (backs up the file as .orig; re-runnable)
"""
import os
import shutil
import b12x.gemm.blockscaled._preparation as m

p = m.__file__
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p + ".orig").read()
old = '''        output_mode="functional" if functional else "provided",
        alpha_mode="unit" if functional else "tensor",
        expected_m=expected_m,
    )
    return _default_lowering(inner, device.identity)
'''
new = '''        output_mode="functional" if functional else "provided",
        alpha_mode="unit" if functional else "tensor",
        expected_m=expected_m,
        **_dense_tile_override(query.out_features, query.padded_in_features, expected_m),
    )
    return _default_lowering(inner, device.identity)


def _dense_tile_override(n, k, expected_m):
    import os as _os_to
    spec = _os_to.environ.get("B12X_DENSE_TILE_OVERRIDE", "")
    if not spec or expected_m is None or expected_m < 2048:
        return {}
    for item in spec.split(","):
        shape, _, tile = item.partition("=")
        sn, sk = (int(v) for v in shape.split("x"))
        if (sn, sk) != (int(n), int(k)):
            continue
        dims, _, flag = tile.partition(":")
        tm, tn, tk = (int(v) for v in dims.split("x"))
        overrides = {"mma_tiler_mn": (tm, tn), "_tile_k_override": tk}
        if flag == "lmu":
            overrides["_large_m_unroll_override"] = True
        return {"overrides": overrides}
    return {}
'''
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched", p)

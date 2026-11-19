"""Patch b12x's DenseGemmKernel persistent tile scheduler to walk tiles along N for narrow outputs.

B12X_DENSE_RASTER_N_MAX_CTAS=<n> (default 0 = stock): when a non-split-K launch has at most <n> tile columns
(N tiles, a static count), the persistent scheduler walks along N instead of M, so the 48 resident CTAs
share one A row-panel while it is in L2 instead of re-streaming A from memory once per N column. With
tile N = 128, n = 20 covers N <= 2,560 (GDN out_proj, attention o_proj, shared-expert gate+up and down).
The value joins DenseGemm compile keys, so no cached stock kernel is reused under the policy.
B12X_DENSE_RASTER_N_MIN_CTAS=<m> (default 1) excludes narrower outputs: shared-expert gate+up (10 tile columns)
got 5% slower in the harness because its 10.5 MB A operand already fits in L2; min = max = 20 limits the
policy to N = 2,560 (GDN out_proj, attention o_proj, shared-expert down).
B12X_DENSE_RASTER_N_LOG=1 prints each decision at compile time.

Usage (in the container): python3 dense_raster_patch.py   (backs up dense_gemm.py as .orig; re-runnable)
"""
import os
import shutil
import b12x._lib.dense_gemm as dg

p = dg.__file__
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p + ".orig").read()

reps = [
    # module-level switch, placed right before the class that owns _compute_grid
    ("class DenseGemmKernel:\n",
     "import os as _os_rn\n"
     "_DENSE_RASTER_N_MAX_CTAS = int(_os_rn.environ.get(\"B12X_DENSE_RASTER_N_MAX_CTAS\", \"0\"))\n"
     "_DENSE_RASTER_N_MIN_CTAS = int(_os_rn.environ.get(\"B12X_DENSE_RASTER_N_MIN_CTAS\", \"1\"))\n"
     "_DENSE_RASTER_N_LOG = _os_rn.environ.get(\"B12X_DENSE_RASTER_N_LOG\", \"0\") == \"1\"\n\n\n"
     "class DenseGemmKernel:\n"),
    # scheduler decision
    ("        cluster_shape_mnl = (1, 1, 1)\n"
     "        tile_sched_params = utils.PersistentTileSchedulerParams(\n"
     "            num_ctas_mnl,\n"
     "            cluster_shape_mnl,\n"
     "            swizzle_size=(\n"
     "                16 if tile_shape_mnk == (128, 128, 64) and not large_m_unroll else 1\n"
     "            ),\n"
     "        )\n",
     "        cluster_shape_mnl = (1, 1, 1)\n"
     "        _n_ctas = num_ctas_mnl[1]\n"
     "        _raster_n = (\n"
     "            _DENSE_RASTER_N_MAX_CTAS > 0\n"
     "            and isinstance(_n_ctas, int)\n"
     "            and _DENSE_RASTER_N_MIN_CTAS <= _n_ctas <= _DENSE_RASTER_N_MAX_CTAS\n"
     "            and not (split_k_slices > 1 or split_k_all_m)\n"
     "            and not direct_one_m_tile_scheduler\n"
     "        )\n"
     "        if _DENSE_RASTER_N_LOG:\n"
     "            print(f\"[b12x dense raster] tile={tile_shape_mnk} n_ctas={_n_ctas} raster_along_n={_raster_n}\", flush=True)\n"
     "        if _raster_n:\n"
     "            tile_sched_params = utils.PersistentTileSchedulerParams(\n"
     "                num_ctas_mnl, cluster_shape_mnl, swizzle_size=1, raster_along_m=False,\n"
     "            )\n"
     "        else:\n"
     "            tile_sched_params = utils.PersistentTileSchedulerParams(\n"
     "                num_ctas_mnl,\n"
     "                cluster_shape_mnl,\n"
     "                swizzle_size=(\n"
     "                    16 if tile_shape_mnk == (128, 128, 64) and not large_m_unroll else 1\n"
     "                ),\n"
     "            )\n"),
    # compile key
    ("        return key if self._weight_only is None else key + (\n"
     "            self._weight_only, self._alpha_reciprocal, self._input_k,\n"
     "        )\n",
     "        if _DENSE_RASTER_N_MAX_CTAS > 0:\n"
     "            key = key + ((\"raster_n_ctas\", _DENSE_RASTER_N_MIN_CTAS, _DENSE_RASTER_N_MAX_CTAS),)\n"
     "        return key if self._weight_only is None else key + (\n"
     "            self._weight_only, self._alpha_reciprocal, self._input_k,\n"
     "        )\n"),
]
for old, new in reps:
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)

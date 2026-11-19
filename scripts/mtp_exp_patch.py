"""Write _cute_prefill_exp.py next to b12x's MTP feedback _cute_prefill.py with env-controlled tile N,
stage count and rasterisation (B12X_MTP_TILE_N, B12X_MTP_STAGES, B12X_MTP_RASTER_N=1 puts the N tiles on
grid x so CTAs that share an A tile run back to back). Serving code is not touched.
Usage (in the container): python3 mtp_exp_patch.py            -> experimental copy _cute_prefill_exp.py
                           python3 mtp_exp_patch.py --inplace  -> patch _cute_prefill.py itself (backup .orig);
  defaults stay stock (16, 3, off), and the variant joins the compile-cache key so cached stock kernels are
  never reused for another variant.
"""
import os
import shutil
import sys
import b12x.sequence.mtp_feedback as m

inplace = "--inplace" in sys.argv
src = os.path.join(os.path.dirname(m.__file__), "_cute_prefill.py")
if inplace and not os.path.exists(src + ".orig"):
    shutil.copy(src, src + ".orig")
if inplace:
    src = src + ".orig"
dst = os.path.join(os.path.dirname(m.__file__), "_cute_prefill.py" if inplace else "_cute_prefill_exp.py")
s = open(src).read()
reps = [
    ("_TILE_N = 16\n", 'import os as _os\n_TILE_N = int(_os.environ.get("B12X_MTP_TILE_N", "16"))\n'
                       '_RASTER_N = _os.environ.get("B12X_MTP_RASTER_N", "0") == "1"\n'),
    ("_STAGES = 3\n", '_STAGES = int(_os.environ.get("B12X_MTP_STAGES", "3"))\n'),
    ("        m_tile, n_tile, _ = cute.arch.block_idx()\n",
     "        if cutlass.const_expr(_RASTER_N):\n"
     "            n_tile, m_tile, _ = cute.arch.block_idx()\n"
     "        else:\n"
     "            m_tile, n_tile, _ = cute.arch.block_idx()\n"),
    ("            grid=(\n                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),\n"
     "                self.output_columns // _TILE_N,\n                1,\n            ),\n",
     "            grid=(\n                self.output_columns // _TILE_N,\n"
     "                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),\n                1,\n"
     "            ) if _RASTER_N else (\n"
     "                (live_rows + Int32(self.tile_m - 1)) // Int32(self.tile_m),\n"
     "                self.output_columns // _TILE_N,\n                1,\n            ),\n"),
    ("                    cache_key[1:],\n",
     "                    cache_key[1:] + ((_TILE_N, _STAGES, _RASTER_N) if (_TILE_N, _STAGES, _RASTER_N) != (16, 3, False) else ()),\n"),
]
if not inplace:
    reps.append(("register_program_cache(_KERNEL_CACHE, lock=_CACHE_LOCK)", "pass  # experimental copy: no shared cache registration"))
for old, new in reps:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(dst, "w").write(s)
print("wrote", dst)

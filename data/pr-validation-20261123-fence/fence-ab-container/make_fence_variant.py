"""Create b12x/sequence/mtp_feedback/_cute_prefill_fence.py from the in-place patched _cute_prefill.py.
B12X_MTP_FENCE selects what precedes each mainloop consumer_release:
  0 = nothing (current kernel), 1 = fence_proxy("async.shared", space="cta") (#9759 fallback for DSL < 13.4),
  2 = inline fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster (#9759 >= 13.4 path).
The fence mode is appended to the compile-cache key."""
import b12x, os, sys
d = os.path.join(os.path.dirname(b12x.__file__), "sequence", "mtp_feedback")
src = open(os.path.join(d, "_cute_prefill.py")).read()
old_rel = "                load_pipeline.consumer_release(consumer_state)\n                consumer_state.advance()\n"
assert src.count(old_rel) == 1, src.count(old_rel)
new_rel = ("                if const_expr(_FENCE == 1):\n"
           "                    cute.arch.fence_proxy(\"async.shared\", space=\"cta\")\n"
           "                if const_expr(_FENCE == 2):\n"
           "                    cute.arch.inline_ptx(\"fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;\")\n"
           + old_rel)
src = src.replace(old_rel, new_rel)
old_cfg = '_STAGES = int(_os.environ.get("B12X_MTP_STAGES", "3"))\n'
assert src.count(old_cfg) == 1
src = src.replace(old_cfg, old_cfg + '_FENCE = int(_os.environ.get("B12X_MTP_FENCE", "0"))\n')
old_key = "cache_key[1:] + ((_TILE_N, _STAGES, _RASTER_N) if (_TILE_N, _STAGES, _RASTER_N) != (16, 3, False) else ()),"
assert src.count(old_key) == 1
src = src.replace(old_key, "cache_key[1:] + (_TILE_N, _STAGES, _RASTER_N, \"fence\", _FENCE),")
out = os.path.join(d, "_cute_prefill_fence.py")
open(out, "w").write(src)
print("wrote", out)

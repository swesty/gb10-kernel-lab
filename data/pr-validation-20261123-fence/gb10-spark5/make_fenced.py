import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = ("                )\n"
       "                load_pipeline.consumer_release(consumer_state)\n"
       "                consumer_state.advance()\n")
assert s.count(old) == 1, s.count(old)
new = ("                )\n"
       "                # The stage was read with ldmatrix (generic proxy) and is refilled by\n"
       "                # TMA (async proxy). The mbarrier arrive alone does not order those\n"
       "                # reads before the refill, so fence them first (same fix as #9759).\n"
       "                if cutlass.const_expr(\n"
       "                    (cutlass.CUDA_VERSION.major, cutlass.CUDA_VERSION.minor) >= (13, 4)\n"
       "                ):\n"
       "                    cute.arch.inline_ptx(\n"
       "                        \"fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;\"\n"
       "                    )\n"
       "                else:\n"
       "                    cute.arch.fence_proxy(\"async.shared\", space=\"cta\")\n"
       "                load_pipeline.consumer_release(consumer_state)\n"
       "                consumer_state.advance()\n")
open(dst, "w").write(s.replace(old, new))
print("ok")

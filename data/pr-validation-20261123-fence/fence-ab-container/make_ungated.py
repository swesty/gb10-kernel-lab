"""Make an ungated copy of the fenced B1 PR file: always emit the release fence PTX, no CUDA 13.4 check."""
import sys
s = open(sys.argv[1]).read()
old = s[s.index("                if cutlass.const_expr(\n                    (cutlass.CUDA_VERSION"):s.index("                load_pipeline.consumer_release(consumer_state)")]
new = ('                cute.arch.inline_ptx(\n'
       '                    "fence.proxy.async::generic.release.sync_restrict::shared::cluster::read.cluster;"\n'
       '                )\n')
assert s.count(old) == 1
open(sys.argv[2], "w").write(s.replace(old, new))
print("ok")

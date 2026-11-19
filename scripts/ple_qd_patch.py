"""Make the PLE disk-table io_uring queue depth configurable (VLLM_PLE_DISK_QUEUE_DEPTH, default 64 = stock).
Usage (in the container): python3 ple_qd_patch.py   (backs up b12x_ple.py as .orig; re-runnable)"""
import os, shutil
import vllm.models.qwen4_exp.nvidia.b12x_ple as m
p = m.__file__
if not os.path.exists(p + ".orig"):
    shutil.copy(p, p + ".orig")
s = open(p + ".orig").read()
old = "            self.disk_table = api.DiskTable(layout, shard_rows)\n"
new = ("            import os as _os_qd\n"
       "            self.disk_table = api.DiskTable(\n"
       "                layout, shard_rows,\n"
       "                queue_depth=int(_os_qd.environ.get(\"VLLM_PLE_DISK_QUEUE_DEPTH\", \"64\")),\n"
       "            )\n")
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched", p)

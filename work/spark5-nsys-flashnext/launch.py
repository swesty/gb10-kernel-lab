import os, sys
d = "/root/.cache/vllm/nsys-flashnext"
env = dict(kv.split("=", 1) for kv in open(d + "/env.bin", "rb").read().decode().split("\0") if "=" in kv)
argv = [a for a in open(d + "/cmdline.bin", "rb").read().decode().split("\0") if a]
serve = argv[1:]  # /usr/local/bin/vllm serve ...
prof = ["--profiler-config", "{\"profiler\":\"cuda\"}"]
mode = sys.argv[1]
if mode == "nsys":
    cmd = ["/opt/nsys/target-linux-sbsa-armv8/nsys", "profile", "-o", d + "/flashnext",
           "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
           "--capture-range=cudaProfilerApi", "--capture-range-end=repeat",
           "--sample=none", "--cpuctxsw=none", "--force-overwrite=true"] + serve + prof
elif mode == "nsysgm":
    cmd = ["/opt/nsys/target-linux-sbsa-armv8/nsys", "profile", "-o", d + "/flashnext-gm",
           "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
           "--capture-range=cudaProfilerApi", "--capture-range-end=repeat",
           "--gpu-metrics-devices=0", "--gpu-metrics-set=file:" + d + "/gb20b-mem.config",
           "--gpu-metrics-frequency=20000",
           "--sample=none", "--cpuctxsw=none", "--force-overwrite=true"] + serve + prof
elif mode == "ncu":
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "3600"
    cmd = ["/usr/local/cuda/bin/ncu", "--target-processes", "all", "--profile-from-start", "off",
           "--graph-profiling", "node", "--kernel-name-base", "demangled",
           "--kernel-name", "regex:wmma_tensorop_bf16|MoEDynamicKernelSilu|libdense_gemmDenseGemmKernel|SelectedPositionPagedForward",
           "--filter-mode", "per-launch-config", "--launch-skip", "2", "--launch-count", "1",
           "--set", "detailed", "--metrics", "lts__d_sectors_fill_device.sum,lts__d_sectors_fill_sysmem.sum",
           "--cache-control", "all", "--clock-control", "none", "--replay-mode", "kernel",
           "-f", "-o", d + "/ncu-flashnext"] + serve + prof
elif mode == "patched":
    env["B12X_SELECTED_PIPELINE"] = "1"
    env["B12X_SELECTED_SPLITQK"] = "1"
    env["B12X_SELECTED_SWZ333"] = "1"
    cmd = serve
elif mode == "patched-compile":
    env["B12X_SELECTED_PIPELINE"] = "1"
    env["B12X_SELECTED_SPLITQK"] = "1"
    env["B12X_SELECTED_SWZ333"] = "1"
    env["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    cmd = serve
elif mode == "dump":
    env["B12X_DUMP_SELECTED"] = d + "/qsa-dump"
    cmd = serve
else:
    cmd = serve
extra = d + "/extra.env"
if os.path.exists(extra):
    for line in open(extra):
        line = line.strip()
        if line.startswith("ARG "):
            # ARG --flag value : replace the value of an existing serve flag, or append it
            _, flag, val = line.split(None, 2)
            if flag in cmd:
                cmd[cmd.index(flag) + 1] = val
            else:
                cmd += [flag, val]
            print("EXTRA ARG:", flag, val, flush=True)
        elif line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
            print("EXTRA ENV:", k, "=", v, flush=True)
print("LAUNCH:", " ".join(cmd), flush=True)
os.execvpe(cmd[0], cmd, env)

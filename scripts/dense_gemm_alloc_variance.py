"""Is the dense MXFP8 GEMM time allocation-dependent? Time the same default kernel on several fresh copies of
the weight (and activation) tensors, interleaved, cold L2. Also records SM clock via NVML around each launch."""
import sys, json, torch
sys.path.insert(0, "/tmp/gemmexp")
from dense_gemm_harness import Problem8, DEV, flush_l2
from b12x._lib.dense_gemm import dense_gemm
from b12x.gemm.blockscaled._linear import _mxfp8_scale_mma_from_input
try:
    import pynvml
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    clk = lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
except Exception:
    clk = lambda: -1

name = sys.argv[1] if len(sys.argv) > 1 else "out"
copies = int(sys.argv[2]) if len(sys.argv) > 2 else 6
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
p = Problem8(name, 4096)
m, n, k = p.m, p.n, p.k
xq, xs = p.quant()
s_a = _mxfp8_scale_mma_from_input(xs, rows=m, width=k, logical_width=k).view(torch.float8_e8m0fnu)
w = p.weight.weight
alpha = torch.ones(1, device=DEV, dtype=torch.float32)
out = torch.empty((m, n, 1), device=DEV, dtype=torch.bfloat16)
pad = []
Ws = []
for i in range(copies):
    pad.append(torch.empty(int(37e6) * (i + 1), dtype=torch.uint8, device=DEV))  # shift placement
    Ws.append((w.values.clone().view(n, k, 1), w.scale_mma.clone().view(torch.float8_e8m0fnu) if w.scale_mma.is_contiguous() else w.scale_mma.view(torch.float8_e8m0fnu)))
As = [(xq.clone().view(m, k, 1), s_a) for _ in range(2)]
def run(a, b):
    dense_gemm(a, b, out, ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16",
               sf_vec_size=32, alpha=alpha, expected_m=4096)
run(As[0], Ws[0]); torch.cuda.synchronize()
res = {}
for r in range(reps):
    for wi, b in enumerate(Ws):
        for ai, a in enumerate(As):
            flush_l2()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); c0 = clk()
            s.record(); run(a, b); e.record(); torch.cuda.synchronize()
            res.setdefault((wi, ai), []).append((round(s.elapsed_time(e) * 1000), c0, clk()))
for key, v in res.items():
    ts = sorted(t for t, _, _ in v)
    print(f"W copy {key[0]} A copy {key[1]}: median {ts[len(ts)//2]} us  min {ts[0]}  max {ts[-1]}  clocks {[c for _, c, _ in v]}")

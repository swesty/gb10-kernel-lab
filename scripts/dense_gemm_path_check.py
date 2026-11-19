"""Resolve why direct dense_gemm sweeps time slower than the live api.mm path for some shapes.
Interleaves, in one process: live api.mm, dense_gemm on the live workspace operands, dense_gemm on harness
operands; and samples the SM clock (NVML) from a thread while each runs in a loop. Prints both lowerings."""
import sys, threading, time, torch
sys.path.insert(0, "/tmp/gemmexp")
from dense_gemm_harness import Problem8, kernel_times, DEV, flush_l2
from b12x._lib.dense_gemm import dense_gemm, _lower_dense_gemm
from b12x._lib.intrinsics import as_grouped_scale_view_mx
from b12x.gemm.blockscaled._linear import _mxfp8_scale_mma_from_input
from b12x.gemm.blockscaled._a16 import _layout
from b12x.gemm.blockscaled._preparation import _HEURISTIC_PLANS
import triton, pynvml

pynvml.nvmlInit(); H = pynvml.nvmlDeviceGetHandleByIndex(0)


def clocked(fn, n):
    samples, stop = [], threading.Event()
    def sampler():
        while not stop.is_set():
            samples.append(pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)); time.sleep(0.01)
    t = threading.Thread(target=sampler); t.start()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record(); torch.cuda.synchronize(); stop.set(); t.join()
    return s.elapsed_time(e) * 1000 / n, (min(samples), max(samples)) if samples else None


name = sys.argv[1] if len(sys.argv) > 1 else "qkvz"
p = Problem8(name, 4096)
m, n, k = p.m, p.n, p.k
ws = p.workspace()
live = lambda: p.api.mm(p.x, p.weight, workspace=ws)
live(); torch.cuda.synchronize()
st = next(d._prepared.state for (dv, q), d in _HEURISTIC_PLANS.items() if getattr(q, "out_features", 0) == n)
sf_start = _layout(m, n, k, False, (64, 64, 1))[0]
size = triton.cdiv(m, 128) * triton.cdiv(k // 32, 4) * 512
A_ws = (ws[: m * k].view(torch.float8_e4m3fn).view(m, k, 1), as_grouped_scale_view_mx(ws[sf_start: sf_start + size].view(1, -1), m, k))
xq, xs = p.quant()
A_my = (xq.view(m, k, 1), _mxfp8_scale_mma_from_input(xs, rows=m, width=k, logical_width=k).view(torch.float8_e8m0fnu))
w = p.weight.weight
B = (w.values.view(n, k, 1), w.scale_mma.view(torch.float8_e8m0fnu))
alpha = torch.ones(1, device=DEV, dtype=torch.float32)
out = torch.empty((m, n, 1), device=DEV, dtype=torch.bfloat16)
kw = dict(ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32, alpha=alpha, expected_m=4096)
low_direct = _lower_dense_gemm(A_my, B, out, **kw).to_dict()
low_live = st.dense.lowering.to_dict()
diff = {kk: (low_live[kk], low_direct[kk]) for kk in low_live if low_live[kk] != low_direct.get(kk)}
print("lowering differences live vs direct:", diff)
cases = {
    "live api.mm": live,
    "dense_gemm ws-operands": lambda: dense_gemm(A_ws, B, out, **kw),
    "dense_gemm my-operands": lambda: dense_gemm(A_my, B, out, **kw),
    "live core.run my-operands": lambda: st.dense.run(A_my, B, out=out, alpha=alpha),
}
for rnd in range(2):
    for label, fn in cases.items():
        fn(); torch.cuda.synchronize()
        kt = kernel_times(fn, 6, cold=True)
        g = sum(v[0] for kk, v in kt.items() if "DenseGemm" in kk)
        hot, clk = clocked(fn, 30)
        print(f"round {rnd} {label:28s} kernel(cold,CUPTI) {g:8.1f} us   loop(events) {hot:8.1f} us/call  SM clock {clk}")

<!-- Draft comment for local-inference-lab/vllm#662 (Qwen3.8-Flash-Next NVFP4 optimization review). Not posted. -->
Prefill results from profiling Qwen3.8-Flash-Next-NVFP4 on a **DGX Spark (GB10, sm_121, TP1)**, in case they are useful
for this checklist. Image `eugr/spark-vllm-b12x@sha256:66147fc1…a41b` (this fork at `ab86b707`, b12x 1.5.0). Each
change sits behind an env switch, was A/B'd live, and matches the engine's own run-to-run noise in teacher-forced
logprobs. Everything (report, patches, harnesses, Nsight reports, a spark-vllm-docker recipe):
https://github.com/swesty/gb10-kernel-lab

| Same session | Stock | All changes |
|---|---|---|
| 14.5K real-text prefill | 6.61 s | 4.76 s (−28%) |
| 16K TTFT | 7,094 ms | 4,892 ms (−31%) |
| Decode | | unchanged |

vLLM-side items, all still present on `dev/karmic-kraken` @ `8244d1c`:

1. **PLE disk table** (`b12x_ple.py`): `api.DiskTable(layout, shard_rows)` uses the default `queue_depth=64`; 256 gives
   −4.1% TTFT. Prefetching the next prefill chunk's rows while the current chunk computes (prediction in
   `model_state.prepare_inputs`, ID-checked, falls back on mismatch) gives another −6.2%. The GPU was otherwise idle
   140–190 ms per chunk waiting on synchronous reads. (b12x side: local-inference-lab/b12x#TBD.)
2. **QSA q/k norm** (`b12x_qsa.py`): `B12xQSAAttention` never calls the existing `fused_qk_rmsnorm_rope_gate`. The eager
   `GemmaRMSNorm` chain + separate mRoPE + `output * sigmoid(gate)` cost ~4 ms per layer per chunk. Fused norm/RoPE plus
   a gate kernel that reads the gate in place: −5% prefill.
3. **GDN identity case** (`qwen_gdn_linear_attn.py`): `_forward_core_b12x_mixed` always gathers inputs, zero-fills and
   scatters, even when the worklist is the identity (plain prefill, no verification rows). Running conv/recurrence/norm
   in place in that case: −4% prefill. #661 may already cover part of this.
4. **HyperConnection**: fusing the up-projection with the gated mean (the T×10,240 gate logits never hit memory) plus a
   one-program-per-token Triton combine-norm for prefill: −5%. Related to, but not the same as, #659.
5. **MXFP8 activation quantisation** fused into the GDN gated norm and the attention output gate (needs a small b12x
   workspace hook): −2%.

Plus b12x kernel issues found along the way (MTP feedback prefill tile, BF16 swizzle in two kernels, dense GEMM tile order),
filed on the b12x repo. Happy to turn any of 1–5 into PRs against whichever branch you prefer.

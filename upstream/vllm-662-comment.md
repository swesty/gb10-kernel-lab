<!-- Draft comment for local-inference-lab/vllm#662 (Qwen3.8-Flash-Next NVFP4 optimization review). Not posted. -->
I have been profiling Qwen3.8-Flash-Next-NVFP4 prefill on a **DGX Spark (GB10, sm_121, TP1)**, and a few
of the findings belong on this checklist. The image was `eugr/spark-vllm-b12x@sha256:66147fc1…a41b` (this fork at
`ab86b707`, b12x 1.5.0). Every change sits behind an environment switch, was A/B tested in the serving engine, and stays
within the engine's own run-to-run noise on teacher-forced logprobs.

| Same session | Stock | All changes |
|---|---|---|
| 14.5K-token real-text prefill | 6.61 s | 4.76 s (−28%) |
| 16K TTFT | 7,094 ms | 4,892 ms (−31%) |
| Decode | | unchanged |

The vLLM-side items below are all still present on `dev/karmic-kraken` @ `8244d1c`:

1. **PLE disk table** (`b12x_ple.py`). `api.DiskTable(layout, shard_rows)` uses the default `queue_depth=64`; 256 cuts TTFT
   by 4.1%. Prefetching the next prefill chunk's rows while the current chunk computes (predicted in
   `model_state.prepare_inputs`, checked against the real IDs, falling back on a mismatch) cuts another 6.2%. Without it,
   the GPU sat idle for 140–190 ms per chunk waiting on synchronous reads. (b12x side: flashinfer-ai/flashinfer#TBD.)
2. **QSA q/k norm** (`b12x_qsa.py`). `B12xQSAAttention` never calls the existing `fused_qk_rmsnorm_rope_gate`. The eager
   `GemmaRMSNorm` chain, the separate mRoPE and `output * sigmoid(gate)` cost about 4 ms per layer per chunk. The fused
   norm and RoPE, plus a gate kernel that reads the gate in place, gave −5% prefill.
3. **GDN identity case** (`qwen_gdn_linear_attn.py`). `_forward_core_b12x_mixed` always gathers its inputs, zero-fills and
   scatters, even when the worklist is the identity (plain prefill with no verification rows). Running conv, recurrence
   and norm in place in that case gave −4% prefill. #661 may already cover part of this.
4. **HyperConnection.** Fusing the up-projection with the gated mean (so the T × 10,240 gate logits never reach memory),
   plus a one-program-per-token Triton combine-norm for prefill, gave −5%. Related to #659, but not the same change.
5. **MXFP8 activation quantization**, fused into the GDN gated norm and the attention output gate. This needs a small b12x
   workspace hook, and gave −2%.

I also found three b12x kernel issues along the way (the MTP feedback prefill tile, the BF16 swizzle in two kernels, and
the dense GEMM tile order) and am filing those with FlashInfer, where b12x now lives. I checked the hardware-dependent
ones on an RTX PRO 6000 too, so the suggested fixes do not regress sm_120.

Everything (the report, patches, harnesses, Nsight reports and a spark-vllm-docker recipe) is at
https://github.com/swesty/gb10-kernel-lab. Happy to turn any of 1 to 5 into PRs against whichever branch you prefer.

<!-- Draft issue for vllm-project/vllm. Not filed. -->
# Qwen4-Exp PLE disk offload: rows are read synchronously, one `preadv` per row, and prefetch is unsupported

**Where:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` on `main` @ `4ae5147`, class
`Qwen4ExpPLEDiskOffloadEmbedding`:
- `stage_rows()` → `_read_rows()`: after a `POSIX_FADV_WILLNEED` pass over the page runs, every unique row is read with a
  serial `os.preadv` from Python, inside the step.
- `start_prefetch()` is a no-op and `finalize_prefetch()` raises "PLE disk offload does not support prefetch". The pinned-host
  backend can overlap its lookup with the preceding decoder layer; the disk backend cannot.

## Why it matters

For Qwen3.8-Flash-Next the n-gram table is ~28.9 GB (320M NVFP4 rows). On a 121 GiB unified-memory DGX Spark it does not fit
next to the model, so disk offload is the practical option. Each 4,096-token prefill chunk looks up 32,768 rows (4,096 tokens ×
8 heads), with a weight plane and a scale plane per row.

We measured this pattern on the vLLM b12x fork (`local-inference-lab/vllm`), which reads the same table through b12x's
`io_uring` `DiskTable`. There the GPU sat **idle 140–190 ms at the start of every prefill chunk** waiting for the read: about
0.7 s of a 16K-token prefill, ~12% of TTFT, and invisible in kernel-time tables. Two changes fixed most of it:
- a deeper `io_uring` queue (64 → 256): −4.1% TTFT;
- prefetching the next chunk's rows in a background thread while the current chunk computes (~1.2 s of GPU time per chunk),
  with an exact row-ID check before use: −6.2% TTFT on top.

**We have not measured mainline's disk backend.** Its serial per-row `preadv` loop in Python, with only readahead hints, is
likely slower than the `io_uring` path we measured, not faster.

## Suggestions

1. Read asynchronously with real queue depth (`io_uring`, or a thread pool issuing `preadv` batches), off the step's critical
   path.
2. Support `start_prefetch` / `finalize_prefetch` for the disk backend. For chunked prefill of a single long prompt, the next
   chunk's tokens are already known in the model runner. Stage the predicted rows for chunk N+1 while chunk N runs, and validate
   them with an exact ID comparison before use, falling back to a normal read on mismatch.

Measurements, design and a working patch (against the fork and b12x, not mainline):
<https://github.com/swesty/gb10-kernel-lab> — `REPORT.md` §13.17, `patches/b12x/ple_prefetch.b12x-and-vllm.diff`,
`scripts/gaps.py` (idle-gap histogram from an Nsight Systems trace).

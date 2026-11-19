<!-- Draft issue for vllm-project/vllm. Not filed. -->
# Qwen4-Exp PLE disk offload: rows are read synchronously, one `preadv` per row, and prefetch is unsupported

**Where:** `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` on `main` @ `4ae5147`, class
`Qwen4ExpPLEDiskOffloadEmbedding`:

- `stage_rows()` calls `_read_rows()`, which issues a `POSIX_FADV_WILLNEED` pass over the page runs and then reads every
  unique row with a serial `os.preadv` from Python, inside the step.
- `start_prefetch()` does nothing, and `finalize_prefetch()` raises "PLE disk offload does not support prefetch". The
  pinned-host backend can overlap its lookup with the preceding decoder layer; the disk backend cannot.

## Why it matters

For Qwen3.8-Flash-Next the n-gram table is about 28.9 GB (320M NVFP4 rows). On a DGX Spark with 121 GiB of unified memory
it does not fit next to the model, so disk offload is the practical option there. Each 4,096-token prefill chunk looks up
32,768 rows (4,096 tokens × 8 heads), with a weight plane and a scale plane per row.

I measured this pattern on the vLLM b12x fork (`local-inference-lab/vllm`), which reads the same table through b12x's
`io_uring` `DiskTable`. There, the GPU sat **idle for 141–186 ms at the start of every prefill chunk** waiting for the
read. That is about 0.7 s of a 16K-token prefill, around 12% of TTFT, and it does not show up in kernel-time tables. Two
changes fixed most of it:

- a deeper `io_uring` queue (64 to 256): −4.1% TTFT;
- prefetching the next chunk's rows in a background thread while the current chunk computes (about 1.2 s of GPU time per
  chunk), with an exact row-ID check before use: another −6.2% TTFT.

I have **not** measured mainline's disk backend. I would expect its serial per-row `preadv` loop in Python, with only
readahead hints, to be slower than the `io_uring` path I measured, but that is an inference from the code.

## Suggestions

1. Read asynchronously with real queue depth (`io_uring`, or a thread pool issuing `preadv` batches), off the step's
   critical path.
2. Support `start_prefetch` and `finalize_prefetch` for the disk backend. For chunked prefill of one long prompt, the next
   chunk's tokens are already known in the model runner. Stage the predicted rows for chunk N+1 while chunk N runs, and
   check them with an exact ID comparison before use, falling back to a normal read on a mismatch.

The measurements, the design and a working patch (against the fork and b12x, not mainline) are at
<https://github.com/swesty/gb10-kernel-lab>: see `REPORT.md` §13.17, `patches/b12x/ple_prefetch.b12x-and-vllm.diff`, and
`scripts/gaps.py` (the idle-gap histogram from an Nsight Systems trace). I can help port it if there is interest.

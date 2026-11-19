<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# PLE `DiskTable`: synchronous disk reads leave the GPU idle at every prefill chunk; proposal for a prefetch API

**Where:** `sequence/_shared/disk_table.py` (`DiskRowCache`, `read_rows`) and `sequence/ple_embedding/_disk.py`
(`DiskTable`, `queue_depth: int = 64`) — present in FlashInfer `main` @ `d3f59f0` (also in the archived standalone local-inference-lab/b12x `master` @ `2cc7f66` / b12x 1.5.0).

## What we saw

Qwen3.8-Flash-Next-NVFP4 with `VLLM_PLE_TABLE_MEMORY=disk` on a DGX Spark. The n-gram table is 28.9 GB (320M NVFP4
rows), too big to keep in memory next to the model. A gap histogram of an Nsight Systems trace showed the GPU **idle
140–190 ms at the start of every 4,096-token prefill chunk**, always between `_hash_ids_kernel` and `_nvfp4_lookup_kernel`.
That is ~0.7 s of a 16K-token prefill (~12% of TTFT), invisible in kernel-time tables.

`DiskTable._run` hashes on the GPU, synchronises, copies 32,768 row IDs to the host, reads ~65K random rows (weight and
scale planes) with `io_uring`, and only then launches the lookup. Nothing is queued on the GPU meanwhile.

## What helped

1. **Queue depth.** `DiskTable(layout, shard_rows, queue_depth=256)` instead of the default 64: TTFT −4.1%
   (1,024 gave no further gain). vLLM currently never passes it.
2. **Prefetch.** While chunk N runs (~1.2 s of GPU time), hash chunk N+1's predicted tokens on the GPU and read their rows
   in a background thread (the native reader already releases the GIL), after an event that guarantees chunk N's lookup
   has finished with the staging buffer. Every real transaction compares its IDs with the resident IDs and reads only
   on a mismatch, so a wrong prediction costs nothing. TTFT −6.2% on top (the first chunk cannot be hidden).

Our implementation adds `read_rows_cached()` / `prefetch()` to `DiskRowCache` and a prefetch mode to `DiskTable._run`;
the prediction lives in vLLM's model state:
<https://github.com/swesty/gb10-kernel-lab/blob/main/patches/b12x/ple_prefetch.b12x-and-vllm.diff>.

## Proposal

A supported API to stage rows for a future batch (IDs or tokens in, rows staged into the next transaction's buffers, with
resident-ID validation), plus a documented `queue_depth` default suited to NVMe. Happy to turn our patch into a PR
against `master`.

Write-up: §13.17 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

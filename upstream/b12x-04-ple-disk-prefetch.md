<!-- Draft issue for flashinfer-ai/flashinfer (experimental/b12x). Not filed. -->
# PLE `DiskTable`: synchronous disk reads leave the GPU idle at the start of every prefill chunk

**Where:** `sequence/_shared/disk_table.py` (`DiskRowCache`, `read_rows`) and `sequence/ple_embedding/_disk.py`
(`DiskTable`, `queue_depth: int = 64`), FlashInfer `main` @ `d3f59f0`. The same code is in the archived standalone
b12x 1.5.0.

I serve Qwen3.8-Flash-Next-NVFP4 on a DGX Spark with `VLLM_PLE_TABLE_MEMORY=disk`, because the n-gram table (28.9 GB,
320M NVFP4 rows) does not fit in memory next to the model. Kernel-time tables showed nothing unusual, but a gap histogram of
an Nsight Systems trace showed the GPU **idle for 141–186 ms at the start of every 4,096-token prefill chunk** (185–191 ms
in the original build), always between `_hash_ids_kernel` and `_nvfp4_lookup_kernel`. The four gaps add up to 697 ms of a
16K-token prefill, roughly 12% of TTFT.

## Why

`DiskTable._run` hashes on the GPU, synchronizes, copies 32,768 row IDs to the host, reads about 65K random rows (a weight
plane and a scale plane per row) with `io_uring`, and only then launches the lookup. Nothing is queued on the GPU in the
meantime.

## What helped

1. **Queue depth.** `DiskTable(layout, shard_rows, queue_depth=256)` instead of the default 64 cut TTFT by 4.1%. Going to
   1,024 gave nothing more. vLLM currently never passes this argument.
2. **Prefetch.** While chunk N runs (about 1.2 s of GPU time), I hash chunk N+1's predicted tokens on the GPU and read their
   rows in a background thread. The native reader already releases the GIL. An event makes sure chunk N's lookup has
   finished with the staging buffer before the reads land. Every real transaction compares its IDs with the resident IDs
   and reads only on a mismatch, so a wrong prediction costs nothing. That cut TTFT by another 6.2%. The first chunk cannot
   be hidden.

My implementation adds `read_rows_cached()` and `prefetch()` to `DiskRowCache` and a prefetch mode to `DiskTable._run`.
The prediction itself lives in vLLM's model state:
<https://github.com/swesty/gb10-kernel-lab/blob/main/patches/b12x/ple_prefetch.b12x-and-vllm.diff>.

## Proposal

A supported way to stage rows for a future batch (IDs or tokens in, rows staged into the next transaction's buffers,
validated against the resident IDs), and a documented `queue_depth` default that suits NVMe. This is host and disk work,
so I would not expect it to behave differently on other GPUs. Happy to turn the patch into a PR.

Write-up: §13.17 of <https://github.com/swesty/gb10-kernel-lab/blob/main/REPORT.md>.

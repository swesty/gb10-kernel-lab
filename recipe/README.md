# Qwen3.8-Flash-Next on a DGX Spark: faster prefill recipe

Source: <https://github.com/swesty/gb10-kernel-lab> (this folder is `recipe/`).

A drop-in variant of eugr's `qwen3.8-flash-next-nvfp4-solo` recipe for
[spark-vllm-docker](https://github.com/eugr/spark-vllm-docker). It adds one mod that installs patched
vLLM and b12x files, and a set of environment switches that turn each optimisation on.

**Measured on one DGX Spark (GB10), same session, stock recipe vs this recipe:**

| | Stock | This recipe | Change |
|---|---|---|---|
| 16K-token prompt, time to first token | 7.09 s | 4.89 s | −31% |
| 14.5K-token real-text prompt, prefill | 6.61 s | 4.76 s | −28% (about 2,200 → 3,060 tokens/s) |
| Decode (1 and 4 concurrent requests) | | | unchanged |

Decode is limited by memory bandwidth and was not changed. The full write-up, including how each change was
found and verified, is `../REPORT.md`.

## Requirements

- One DGX Spark / GB10 (sm_121). Not tested on other SM12x parts.
- A spark-vllm-docker checkout (tested at commit `bb6ee76`).
- **The exact tested image.** The patches are whole files written against one build (vLLM
  `0.1.dev21553+gab86b7073`, b12x 1.5.0). The mod checks every file it replaces and refuses to run on
  anything else, so a newer `eugr/spark-vllm-b12x:latest` will not work:
  ```bash
  docker pull eugr/spark-vllm-b12x@sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b
  docker tag  eugr/spark-vllm-b12x@sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b vllm-node-b12x:latest
  ```
  If you already have a different `vllm-node-b12x:latest`, retag it first (for example as
  `vllm-node-b12x:previous`) so you can go back.
- The model `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (tested snapshot `6909a5bed089`). Its 28.9 GB
  n-gram table is read from disk during prefill, so NVMe speed affects the results.

## Install

From this folder:
```bash
cp recipes/qwen3.8-flash-next-nvfp4-solo-gb10fast.yaml  <spark-vllm-docker>/recipes/
cp -r mods/flashnext-gb10-fast                          <spark-vllm-docker>/mods/
```

## Run

```bash
cd <spark-vllm-docker>
./run-recipe.sh qwen3.8-flash-next-nvfp4-solo-gb10fast --solo --dry-run   # check: --apply-mod mods/flashnext-gb10-fast
./run-recipe.sh qwen3.8-flash-next-nvfp4-solo-gb10fast --solo             # add --port, -d as usual
```
Because `vllm-node-b12x:latest` already exists, the recipe uses it and does not pull or build. The container
log should show `[flashnext-gb10-fast] installed 13 files` before vLLM starts. Startup takes ~3–4 minutes once
b12x's kernel caches are warm. The very first start of the model (stock recipe or this one) autotunes
b12x kernels and can take much longer; leave the machine otherwise idle while it does, because it needs
memory headroom.

## Check it

```bash
bench/bench.sh 8000      # port of your engine
```
Run it once with the stock recipe (`qwen3.8-flash-next-nvfp4-solo`, same image) and once with this one.
"Mean TTFT" on the 16K prefill is the number to compare: ~7.1 s → ~4.9 s on the test machine. Decode TPOT
should be the same.

## What it changes

Every change is behind its own switch in the recipe's `env:` block; delete a line to turn that change off.
With all switches removed, the patched files behave like the stock ones.

| Switch(es) | What it does | Prefill gain alone* |
|---|---|---|
| `B12X_SELECTED_PIPELINE`, `_SPLITQK`, `_SWZ333` | QSA selected-position attention: overlapped loads, QK and softmax on all warps, fixed shared-memory swizzle (1.9× kernel) | −4.5% |
| `VLLM_HC_FUSED_GATE`, `VLLM_HC_TRITON_COMBINE_NORM` | HyperConnection: up-projection fused with the gated mean; faster combine-norm | −5% |
| `VLLM_GDN_IDENTITY_FASTPATH` | GDN: skip gather/scatter copies when no verification rows are present | −4% |
| `VLLM_QSA_FUSED_QK_NORM` | Attention: fused q/k RMSNorm + mRoPE, gate read in place | −5% |
| `B12X_MTP_TILE_N`, `_STAGES`, `_RASTER_N` | MTP feedback projection: fixes a tile shape that re-read 13 GB per chunk (22× kernel) | −3.7% |
| `B12X_DENSE_RASTER_N_MIN/MAX_CTAS`, `B12X_DENSE_TILE_OVERRIDE` | Dense MXFP8 GEMMs: better tile order and tile shape for the affected shapes | −2% |
| `VLLM_PLE_DISK_QUEUE_DEPTH`, `VLLM_PLE_PREFETCH` | n-gram table on disk: deeper `io_uring` queue; read the next prefill chunk's rows while the current one computes | −7 to −10% |
| `VLLM_GDN_FUSED_NORM_QUANT`, `VLLM_QSA_FUSED_GATE_QUANT` | MXFP8 activation quantisation fused into the kernels that produce the activations | −1.5 to −2% |

\* Measured one at a time, each on top of the rows above it, on the 14.5K real-text and 16K random prompts
(ranges cover both); see `../REPORT.md` §9–§13.

## Things to know

- **Unofficial.** These are local patches, not upstream vLLM or b12x. Tested on one machine.
- **Output.** Several changes are bit-identical (the MTP fix, the dense GEMM changes, the PLE changes, the
  attention gate fusion). The others change rounding in a small fraction of values: the attention kernel's
  summation order (max error 1e-4 against FP32, same as stock), RoPE rounding order in the fused q/k norm,
  sigmoid rounding in combine-norm, and the GDN norm fusion (5 of 25 million FP8 activation values, one
  step apart). Every change was checked against the engine's own run-to-run noise with teacher-forced
  log-probabilities. The engine itself is not deterministic (its MoE accumulates in BF16 atomics), so text
  differs run to run with or without this recipe.
- **PLE prefetch** only predicts ahead when a single request is prefilling; with mixed batches it simply
  reads as before. Correctness never depends on the prediction.
- **Memory.** No meaningful extra memory (a 256 KB pinned buffer for prefetched row IDs).
- **Newer images.** The mod refuses to patch them. To port, apply the diffs in `../patches/` by hand.
- **Licence.** The files under `mods/flashnext-gb10-fast/files/` are modified copies of Apache-2.0 vLLM and
  b12x sources (plus one new file); the changes are documented in `../patches/` and `../REPORT.md`.

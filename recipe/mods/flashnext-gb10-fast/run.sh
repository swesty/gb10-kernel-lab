#!/bin/bash
# flashnext-gb10-fast: install the GB10 prefill optimisations for Qwen3.8-Flash-Next-NVFP4 into a fresh
# vllm-node-b12x container. Every change is behind an environment switch (set by the recipe); with the
# switches unset the patched files behave like the stock ones.
#
# The patches are whole files made against one exact image:
#   eugr/spark-vllm-b12x@sha256:66147fc1420dbd59a7bb1411697d81a8231c298105b61c5962179cd146b4a41b
#   (vLLM 0.1.dev21553+gab86b7073, b12x 1.5.0). Before touching anything this script checks that every file
# it replaces is byte-identical to that image's version, and refuses to run otherwise.
set -euo pipefail
cd "$(dirname "$0")"

SITE=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "[flashnext-gb10-fast] site-packages: $SITE"

bad=0
while read -r want rel; do
    [ -n "${rel:-}" ] || continue
    target="$SITE/$rel"
    if [ ! -f "$target" ]; then
        echo "[flashnext-gb10-fast] missing $rel"; bad=1; continue
    fi
    have=$(sha256sum "$target" | cut -d' ' -f1)
    patched=$(sha256sum "files/$rel" | cut -d' ' -f1)
    if [ "$have" != "$want" ] && [ "$have" != "$patched" ]; then
        echo "[flashnext-gb10-fast] $rel differs from the tested image (sha256 $have)"; bad=1
    fi
done < SHA256SUMS.stock
if [ "$bad" -ne 0 ]; then
    echo "[flashnext-gb10-fast] This container is not the tested image; refusing to patch." >&2
    echo "[flashnext-gb10-fast] Use eugr/spark-vllm-b12x@sha256:66147fc1...a41b, or port patches/ by hand." >&2
    exit 1
fi

while read -r sum rel; do
    [ -n "${rel:-}" ] || continue
    install -D -m 0644 "files/$rel" "$SITE/$rel"
    [ "$(sha256sum "$SITE/$rel" | cut -d' ' -f1)" = "$sum" ] || { echo "copy check failed: $rel" >&2; exit 1; }
done < SHA256SUMS.patched
echo "[flashnext-gb10-fast] installed $(grep -c . SHA256SUMS.patched) files"

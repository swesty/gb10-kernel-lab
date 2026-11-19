cd /usr/local/lib/python3.12/dist-packages/b12x
for c in SelectedPositionPagedForward PackedCombineNorm RepresentativeScore PackedRecurrentQwen; do
  echo "== $c"; grep -rlE "$c" --include=*.py . | head -4
done
F=$(grep -rlE "class SelectedPositionPagedForward" --include=*.py . | head -1); [ -z "$F" ] && F=$(grep -rlE "SelectedPositionPagedForward" --include=*.py . | head -1)
echo "### paged attention source: $F ($(wc -l < $F) lines)"
grep -nE -i "swizzle|make_swizzle|Swizzle\(|padding|pad|smem_layout|make_layout|composition|ldmatrix|LdMatrix|cp_async|CopyG2S|tma|stmatrix|bank" $F | head -40

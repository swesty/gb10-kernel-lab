cd /usr/local/lib/python3.12/dist-packages
echo "--- Triton-style kernel definitions"
for k in _gate_mean_kernel _gated_rmsnorm_kernel _sorted_softmax_topk_kernel _causal_conv1d_update_kernel; do
  f=$(grep -rl "def $k" --include=*.py vllm b12x 2>/dev/null | head -1); echo "$k -> $f"; [ -n "$f" ] && grep -n -B3 "def $k" "$f" | grep -E "@|jit" | head -2
done
echo "--- b12x kernel classes"
for c in SelectedPositionPagedForwardKernel PackedCombineNorm RepresentativeScoreKernel PackedRecurrentQwenKernel MoEDynamicKernelSilu DenseGemmKernel; do
  f=$(grep -rlE "class $c\b" --include=*.py b12x 2>/dev/null | head -1); echo "$c -> $f"
done
echo "--- DSL imports in b12x"
grep -rhoE "^(import cutlass[a-zA-Z._]*|from cutlass[a-zA-Z._]* import)" --include=*.py b12x | sort | uniq -c | sort -rn | head -5
echo "--- b12x version"; python3 -c "import b12x,importlib.metadata as m; print(m.version('b12x'))" 2>/dev/null

cd /usr/local/lib/python3.12/dist-packages
echo "### hyperconnection module"; ls b12x/norm/hyperconnection/
grep -n "^def \|^class \|@triton.jit\|\"\"\"" b12x/norm/hyperconnection/_kernels.py | head -30
echo "### _gate_mean_kernel"; N=$(grep -n "def _gate_mean_kernel" b12x/norm/hyperconnection/_kernels.py | cut -d: -f1); sed -n "$((N-2)),$((N+30))p" b12x/norm/hyperconnection/_kernels.py
echo "### who calls gate_mean / combine norm in the model"
grep -rn "gate_mean\|combine_norm\|PackedCombineNorm\|hyperconnection" --include=*.py vllm/model_executor/models/qwen4_exp* 2>/dev/null | head -15
echo "### compilation mode forcing"
grep -rn "CompilationMode.NONE\|enforce_eager\|mode = CompilationMode" --include=*.py vllm/model_executor/models/qwen4_exp* vllm/model_executor/warmup/b12x*.py b12x/integration 2>/dev/null | head -10

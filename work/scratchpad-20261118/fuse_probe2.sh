cd /usr/local/lib/python3.12/dist-packages
echo "### callers"; grep -rln "gate_mean\|combine_norm\|hyperconnection" --include=*.py vllm | head -8
F=$(grep -rln "gate_mean" --include=*.py vllm | head -1); echo "### in $F"; grep -n "gate_mean\|combine_norm\|def forward\|hyperconnection\|engram\|streams" $F | head -30
echo "### b12x api"; grep -n "^def \|gate_mean\|combine_norm" b12x/norm/hyperconnection/api.py | head -30
echo "### compile mode"; grep -rn "CompilationMode.NONE" --include=*.py vllm/config vllm/platforms 2>/dev/null | head -5; grep -rn "b12x" --include=*.py vllm/config/compilation.py vllm/config/vllm.py 2>/dev/null | head -8

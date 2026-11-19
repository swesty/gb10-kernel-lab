cd /usr/local/lib/python3.12/dist-packages
F=b12x/attention/paged/_selected_forward.py
echo "### imports/constants"; sed -n 1,58p $F | grep -nE "import|_SELECTION_WIDTH|_HEAD_DIM"
grep -rn "_SELECTION_WIDTH *=" --include=*.py b12x | head -3
echo "### _cache_key/_compile"; sed -n 1150,1316p $F
echo "### selected position handling in kernel"; sed -n 355,425p $F
echo "### callers"; grep -rn "launch_selected_paged_gqa_direct\|launch_sparse_gqa_split" --include=*.py vllm b12x | grep -v "def launch" | head -8

cd /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels
grep -nE "^(from|import)|class |def " silu.py | head -20
echo "--- files in kernels dir"; ls
echo "--- swizzle / smem layout usage across moe/_shared"
grep -rnE -i "swizzle|get_smem_layout_atom|make_smem_layout_atom|LdMatrix|StMatrix|st_shared|cp_async|make_ordered_layout|padding" --include=*.py .. | cut -c1-170 | head -40

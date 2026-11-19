cd /usr/local/lib/python3.12/dist-packages/b12x
echo "### files using TMA/mbarrier"
grep -rlE "make_tiled_tma_atom|CopyBulkTensorTile|cp\.async\.bulk\.tensor|tma_partition|mbarrier" --include=*.py . | head -20
echo "### attention files using TMA"
grep -rlE "make_tiled_tma_atom|CopyBulkTensorTile|cp\.async\.bulk\.tensor" --include=*.py attention | head
echo "### raw PTX bulk tensor helpers in intrinsics"
grep -nE "def .*(tma|bulk|mbarrier|mbar)" _lib/intrinsics.py | head -30

cd /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe
F=_impl.py
echo "### _dynamic_tile_mn_override callers"; grep -n "_dynamic_tile_mn_override()\|def _select_dynamic_tile\|def _dynamic_tile_mn\b\|_DYNAMIC_TILE_MN_OVERRIDE =" $F | head
echo "### function containing B12X_MOE_TILE_MN"; N=$(grep -n "B12X_MOE_TILE_MN" $F | head -1 | cut -d: -f1); awk -v n=$N 'NR<=n && /^def /{l=NR": "$0} END{}; NR==n{print l}' $F; sed -n "$((N-8)),$((N+22))p" $F
echo "### callers of that function"; FN=$(awk -v n=$N 'NR<=n && /^def /{name=$2} NR==n{print name}' $F | cut -d"(" -f1); echo "fn=$FN"; grep -n "$FN(" $F | head
echo "### supported dynamic tiles"; grep -n "SUPPORTED.*TILE\|_DYNAMIC_TILES\|tile_mn not in\|valid tile\|(64, 128)\|(128, 64)\|(128, 256)" $F | head -15

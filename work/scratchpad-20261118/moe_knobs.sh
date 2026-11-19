cd /usr/local/lib/python3.12/dist-packages/b12x
for k in B12X_DYNAMIC_TILE_MN B12X_MOE_TILE_MN B12X_DYNAMIC_WORK_SOURCE B12X_DYNAMIC_MAX_ACTIVE_CLUSTERS B12X_DYNAMIC_ENABLE_MULTICTA B12X_DYNAMIC_SWAP_AB B12X_MICRO_DYNAMIC_CUTOVER_PAIRS; do
  echo "##### $k"; grep -rn -B2 -A8 "\"$k\"" --include=*.py moe integration 2>/dev/null | grep -vE "^\s*$" | head -16 | cut -c1-180
done

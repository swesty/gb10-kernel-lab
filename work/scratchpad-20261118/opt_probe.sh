cd /usr/local/lib/python3.12/dist-packages
echo "--- b12x metadata"; python3 - <<'P'
import importlib.metadata as m
d = m.metadata("b12x"); print("version", d["Version"]); print("license", d.get("License"), d.get("License-Expression")); print("home", d.get("Home-page")); [print("url", u) for u in (d.get_all("Project-URL") or [])]
P
ls b12x-*.dist-info 2>/dev/null; cat b12x-*.dist-info/direct_url.json 2>/dev/null; echo
echo "--- CuTe DSL env knobs"; grep -rhoE "CUTE_DSL_[A-Z_]+" --include=*.py nvidia_cutlass_dsl cutlass 2>/dev/null | sort -u | head -30
echo "--- attention tunables"; F=b12x/attention/paged/_selected_forward.py
grep -nE "tile_m|tile_n|num_warps|num_threads|_HEAD_DIM =|def __init__|B12X_[A-Z_]+" $F | head -25
grep -rhoE "B12X_[A-Z_]+" --include=*.py b12x | sort -u | head -40
echo "--- tests/bench shipped?"; ls b12x | head -40; ls b12x/tests b12x/bench* 2>/dev/null | head

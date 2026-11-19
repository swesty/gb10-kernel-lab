cd /usr/local/lib/python3.12/dist-packages/b12x
echo "### MoE env knobs"; grep -rhoE "B12X_[A-Z0-9_]*" --include=*.py moe integration | sort -u | grep -iE "MOE|DYNAMIC|MICRO|PHASE|POLICY|TILE|PREFILL|DECODE|BACKEND|KERNEL" | head -40
echo "### where dynamic vs micro/phase is chosen"
grep -rn "MoEDynamicKernel\|MoEMicroKernel\|nvfp4_phase1\|def select\|def choose\|def plan_\|policy" --include=*.py moe/_shared/execution.py moe/*.py 2>/dev/null | head -30

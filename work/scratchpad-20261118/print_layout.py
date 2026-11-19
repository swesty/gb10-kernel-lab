import cutlass, cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90
from cutlass import BFloat16
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum
print("atom enum:", sm90.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, 256))
@cute.jit
def show():
    atom = warpgroup.make_smem_layout_atom(sm90.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, BFloat16, 256), BFloat16)
    print("atom:", atom)
    key = cute.tile_to_shape(atom, (32, 256), order=(0, 1))
    print("key layout:", key)
    for r in range(9):
        print("row", r, "col0 ->", cute.crd2idx((r, 0), key), " col8 ->", cute.crd2idx((r, 8), key), " col64 ->", cute.crd2idx((r, 64), key))
show()

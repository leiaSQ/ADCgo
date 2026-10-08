import sys
sys.path.append('/home/hd/hd_hd/hd_hh323/adcgen')
from adcgen import Operators, GroundState, IntermediateStates, SecularMatrix

h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="ip")
m = SecularMatrix(isr)

for b1 in ["h", "2h1p", "3h2p"]:
    for b2 in ["h", "2h1p", "3h2p"]:
        try:
            expr = m.isr_matrix_block(order=1, block=f"{b1},{b2}")
            print(f"Block {b1},{b2} works")
        except Exception as e:
            print(f"Block {b1},{b2} failed: {e}")

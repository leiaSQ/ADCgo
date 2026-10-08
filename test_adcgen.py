import sys
sys.path.append('/home/hd/hd_hd/hd_hh323/adcgen')
from adcgen import Operators, GroundState, IntermediateStates, SecularMatrix

h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="ip")
m = SecularMatrix(isr)

print("Blocks:")
print(m.valid_blocks)

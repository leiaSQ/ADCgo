import sys
sys.path.append('/home/hd/hd_hd/hd_hh323/adcgen')
from adcgen import Operators, GroundState, IntermediateStates, SecularMatrix

h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="ip")
m = SecularMatrix(isr)

expr = m.isr_matrix_block(order=2, block="1h,2h1p", indices="i,jka")
print("1h,2h1p succeeded")

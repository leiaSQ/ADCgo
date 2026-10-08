import sys
sys.path.append('/home/hd/hd_hd/hd_hh323/adcgen')
from adcgen import Operators, GroundState, IntermediateStates, SecularMatrix

h = Operators(variant="mp")
mp = GroundState(h, first_order_singles=False)
isr = IntermediateStates(mp, variant="ip")
m = SecularMatrix(isr)

expr = m.isr_matrix_block(order=2, block="h,2h1p", indices="i,jka")
print("h,2h1p succeeded")
expr2 = m.isr_matrix_block(order=2, block="2h1p,3h2p", indices="ija,klmcd")
print("2h1p,3h2p succeeded")

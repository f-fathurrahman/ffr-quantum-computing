import math
import random
from qc4p import ops, state

Rz = ops.RotationZ(math.pi)
Rz.dump("RotationZ pi/2")

S = ops.Sgate()
S.dump("S gate")


psi = state.qubit(random.random())
psi.dump("Random state")

S(psi).dump("After S-gate")
Rz(psi).dump("After Rz-gate")

S(S(psi)).dump("After S-gate two times:")

Z = ops.PauliZ()
Z(psi).dump("After Z-gate")

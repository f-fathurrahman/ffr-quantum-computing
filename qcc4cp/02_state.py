import numpy as np
from qc4p import state
from qc4p import helper

ket0 = state.zeros(1)
ket1 = state.ones(1)

# Initialize a state
ψ = 0.1*ket0 + 0.3*ket1
ψ.normalize() # need to do this
state.dump_state(ψ)
# TODO:
# Check the normalization: np.dot(ψ, ψ)
# Check np.dot(ket0, ψ)


# Using tensor product
ket00 = ket0 * ket0
state.dump_state(ket00)

ket10 = ket1 * ket0 # |10>
state.dump_state(ket10)

psi = 0.1*ket00 - 0.1j*ket10
psi.normalize()
state.dump_state(psi)
# Amplitude 
print(psi.ampl(1, 0))
print(psi.prob(1, 0))
#
for bits in helper.bitprod(psi.nbits):
    print("bits = ", bits, " prob = ", psi.prob(*bits))


# From bit string
ket110 = state.bitstring(1, 1, 0)
state.dump_state(ket110)



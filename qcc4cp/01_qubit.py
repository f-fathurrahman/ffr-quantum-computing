import numpy as np
from qc4p import state

q0 = state.qubit(alpha=0.1)
print(q0)
state.dump_state(q0)


q1 = state.qubit(alpha=0.2)
print(q1)
state.dump_state(q1)

# Define some basis

# computational basis
ket0 = state.zeros(1)
ket1 = state.ones(1)

# Hadamard basis
ketPlus = state.plus()
ketMinus = state.minus()

# ??? the name?
ketPlusI = state.plusi()
ketMinusI = state.minusi()

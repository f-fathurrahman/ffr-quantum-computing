from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
test_g = QuantumCircuit(q, name="qc")

test_g.h(q[0])
test_g.x(q[1])
test_g.cz(q[0], q[1])
test_g.x(q[1])

import our_qiskit_functions as oqf

print("\nInitial state")
oqf.Wavefunction(test_g)

f = oqf.blackbox_g_D(test_g, q)

print("\nAfter blackbox")
oqf.Wavefunction(test_g)


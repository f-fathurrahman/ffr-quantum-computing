from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

import our_qiskit_functions as oqf

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
qc = QuantumCircuit(q, name="qc")

# |00>
qc.id(q[0])
qc.id(q[1])

# |01>
#qc.id(q[0])
#qc.x(q[1])

# |10>
#qc.x(q[0])
#qc.id(q[1])

# |11>
#qc.x(q[0])
#qc.x(q[1])

print("Initial: ")
oqf.Wavefunction(qc)

qc.x(q[0])
qc.cx(q[0], q[1])
qc.x(q[0])

print("After: ")
oqf.Wavefunction(qc)

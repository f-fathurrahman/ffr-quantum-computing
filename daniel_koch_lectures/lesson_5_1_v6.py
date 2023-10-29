from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math
import our_qiskit_functions as oqf

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
c = ClassicalRegister(2, name="c")
qc = QuantumCircuit(q, c, name="qc")

qc.h(q[0])
qc.x(q[1])
qc.h(q[1])

print("Initial state: ")
oqf.Wavefunction(qc)

# Call blackbox fuction
f = oqf.blackbox_g_D(qc, q)

print("After blackbox_g_D:")
oqf.Wavefunction(qc)

# Apply H2 again
qc.h(q[0])
qc.h(q[1])

print("After 2nd H2:")
oqf.Wavefunction(qc)


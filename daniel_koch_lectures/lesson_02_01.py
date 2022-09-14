from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute

import numpy as np

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(3, name="q")
c = ClassicalRegister(3, name="c")
super0 = QuantumCircuit(q, c, name="qc")

super0.h(q[0])
super0.id(q[1])
super0.id(q[2])

super0.measure(q[0], c[0])

print(super0.qasm())


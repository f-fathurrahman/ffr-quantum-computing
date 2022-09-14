from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute

import numpy as np

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(3, name="q")
c = ClassicalRegister(3, name="c")
two_q = QuantumCircuit(q, c, name="qc")

two_q.h(q[0])
two_q.id(q[1])

two_q.measure(q[0], c[0])

str_instructions = two_q.qasm()

print("\n-------- data ---------")
print(two_q.data)

print("\n-------- qregs ---------")
print(two_q.qregs)

print("\n-------- cregs ---------")
print(two_q.cregs)

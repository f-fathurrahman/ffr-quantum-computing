# Partial measurement, 2 qubits

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2)
c = ClassicalRegister(2)
qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.h(q[1])
qc.measure(q[0], c[0])
#qc.measure(q[0], c[1])

job = execute(qc, M_simulator)
result = job.result()
rescnt = result.get_counts(qc)
print("rescnt = ", rescnt)

# Using 3 qubit

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(3)
three_qubits = QuantumCircuit(q)

three_qubits.id(q[0])
#three_qubits.x(q[0])
#three_qubits.id(q[1])
three_qubits.x(q[1])
three_qubits.id(q[2])

job = execute(three_qubits, S_simulator)
result = job.result()

print(result.get_statevector())

import our_qiskit_functions as oqf
oqf.Wavefunction(np.asarray(result.get_statevector()))


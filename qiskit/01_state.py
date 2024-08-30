from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

q = QuantumRegister(1)
qc = QuantumCircuit(q)
#qc.id(q[0])
#qc.x(q[0]) # Pauli X
#qc.y(q[0]) # Pauli Y
qc.z(q[0]) # Pauli Z
#qc.h(q[0]) # Hadamard

backend = Aer.backends(name="statevector_simulator")[0]
tqc = transpile(qc, backend)
job = backend.run(tqc)
result = job.result()
psi = result.get_statevector()
print("psi = ", psi)


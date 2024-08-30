from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(1)

#qc.id(0)
#qc.x(0) # Pauli X
#qc.y(0) # Pauli Y
#qc.z(0) # Pauli Z
qc.h(0) # Hadamard

backend = Aer.get_backend("statevector_simulator")
tqc = transpile(qc, backend)
job = backend.run(tqc)
result = job.result()
psi = result.get_statevector(tqc)
print("psi = ", psi)


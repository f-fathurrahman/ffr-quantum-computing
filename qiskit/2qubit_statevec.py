from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(2)
# qubit: | q1 q0 >

qc.h(0) # Hadamard
#qc.h(1) # Hadamard on q1

backend = Aer.get_backend("statevector_simulator")
tqc = transpile(qc, backend)
job = backend.run(tqc) # no need for shots kwarg
result = job.result()
psi = result.get_statevector(tqc)
print("psi = ", psi)


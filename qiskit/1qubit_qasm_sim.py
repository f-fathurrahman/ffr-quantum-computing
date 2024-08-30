from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(1)

#qc.id(0)
#qc.x(0) # Pauli X
#qc.y(0) # Pauli Y
#qc.z(0) # Pauli Z
qc.h(0) # Hadamard

qc.measure_all() # need this for qasm_simulator

backend = Aer.get_backend("qasm_simulator")
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=1000)
result = job.result()
counts = result.get_counts(tqc)
print("counts = ", counts)


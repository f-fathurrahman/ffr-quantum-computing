from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

qc = QuantumCircuit(2)
# qubit: | q1 q0 >

qc.h(0) # Hadamard
#qc.h(1) # Hadamard on q1

qc.measure_all() # need this for qasm_simulator

backend = Aer.get_backend("qasm_simulator")
tqc = transpile(qc, backend)
job = backend.run(tqc, shots=1000)
result = job.result()
counts = result.get_counts(tqc)
print("counts = ", counts)


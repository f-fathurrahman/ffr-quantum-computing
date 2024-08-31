from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import Aer

def run_on_qasm_simulator(my_qc, shots=1000):
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc, shots=shots)
    return job.result()

# Placeholder variables
theta = ParameterVector("θ", 3)
qc = QuantumCircuit(3)
qc.h([0,1,2])
qc.p(theta[0], 0)
qc.p(theta[1], 1)
qc.p(theta[2], 2)
qc.draw()

from math import pi
theta_vals = [pi/8, pi/4, pi/2]
b_qc = qc.assign_parameters(theta_vals)
b_qc.measure_all()

res = run_on_qasm_simulator(b_qc)


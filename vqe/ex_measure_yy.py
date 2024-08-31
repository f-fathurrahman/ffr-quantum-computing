from qiskit import QuantumCircuit

from qiskit import transpile
from qiskit_aer import Aer
def run_on_qasm_simulator(my_qc, shots=1000):
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc, shots=shots)
    return job.result()

qc = QuantumCircuit(2, 1)
# Prepare something here
qc.h([0, 1])
#qc.x(0)
#qc.x(1)
qc.barrier()
# Now the circuit to measure
qc.sdg(0)
qc.sdg(1)
qc.h(0)
qc.h(1)
qc.cx(0,1)
qc.measure(1, 0) # q1 -> c0

import matplotlib.pyplot as plt
qc.draw("mpl")
plt.savefig("IMG_measure_YY.png", dpi=150)

res = run_on_qasm_simulator(qc)
print(res.get_counts())


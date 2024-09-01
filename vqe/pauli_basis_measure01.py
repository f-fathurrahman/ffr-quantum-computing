from qiskit.circuit import QuantumCircuit

from qiskit import transpile
from qiskit_aer import Aer
def run_on_qasm_simulator(my_qc, shots=1000):
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc, shots=shots)
    return job.result()

def run_on_statevec_sim(my_qc):
    backend = Aer.get_backend("statevector_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc)
    return job.result()

# create a circuit, where we would like to measure
# q0 in the X basis, q1 in the Y basis and q2 in the Z basis
qc = QuantumCircuit(3)
qc.ry(0.8, 0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.barrier()
 
# diagonalize X with the Hadamard gate 
qc.h(0)
 
# diagonalize Y with Hadamard as S^\dagger
qc.sdg(1)
qc.h(1)
 
# the Z basis is the default, no action required here
 
# measure all qubits
#qc.measure_all()

#import matplotlib.pyplot as plt
#qc.draw("mpl")
#plt.savefig("IMG_pauli_measure01.png", dpi=150)

#counts = run_on_qasm_simulator(qc).get_counts()
#print(counts)



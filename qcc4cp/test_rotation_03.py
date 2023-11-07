import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

plt.close()
plt.clf()
v0 = Statevector([1/np.sqrt(2), 1j/np.sqrt(2)])
v0.draw("bloch")
plt.savefig("IMG_initial.png", dpi=150)

qc = QuantumCircuit(1)
qc.rz(np.pi/4, 0)

v1 = v0.evolve(qc)

plt.close()
plt.clf()
v1.draw("bloch")
plt.savefig("IMG_final.png", dpi=150)


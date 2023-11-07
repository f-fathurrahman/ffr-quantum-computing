import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

plt.close()
plt.clf()
v0 = Statevector([1/np.sqrt(2), 1j/np.sqrt(2)])
v0.draw("bloch")
plt.savefig("IMG_0.png", dpi=150)

Δθ = 20
for i in range(1,19):
    qc = QuantumCircuit(1)
    qc.rz(np.deg2rad(i*Δθ), 0)
    v1 = v0.evolve(qc)
    plt.close()
    plt.clf()
    v1.draw("bloch")
    plt.savefig("IMG_" + str(i) + ".png", dpi=150)


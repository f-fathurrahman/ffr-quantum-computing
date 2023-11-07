import numpy as np
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector

plt.close()
plt.clf()
α = 0.5
β = np.sqrt(1 - α**2)
v0 = Statevector([α, β])
v0.draw("bloch")
plt.savefig("IMG_0.png", dpi=150)

Δθ = 20
for i in range(1,37):
    qc = QuantumCircuit(1)
    qc.ry(np.deg2rad(i*Δθ), 0)
    v1 = v0.evolve(qc)
    plt.close()
    plt.clf()
    v1.draw("bloch")
    plt.savefig("IMG_" + str(i) + ".png", dpi=150)
    print(f"i = {i} is done")

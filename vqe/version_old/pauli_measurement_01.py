from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.legacy import pauli_measurement

qreg = QuantumRegister(4)
creg = ClassicalRegister(4)
qc = QuantumCircuit(qreg, creg)
pauli_measurement(qc, Pauli('XYZX'), qreg , creg, barrier=True)

import matplotlib.pyplot as plt
qc.draw("mpl")
plt.savefig("IMG_pauli_XYZX.png", dpi=150)

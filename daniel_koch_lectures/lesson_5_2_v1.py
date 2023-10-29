from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import math
import our_qiskit_functions as oqf

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(3, name="q")
anc = QuantumRegister(1, name="anc")
DJ_qc = QuantumCircuit(q, anc, name="qc")

DJ_qc.h(q[0])
DJ_qc.h(q[1])
DJ_qc.h(q[2])
DJ_qc.x(anc[0])

oqf.Wavefunction(DJ_qc, systems=[3,1], show_systems=[True, False])

DJ_qc.h(anc[0])

f = oqf.blackbox_g_DJ(3, DJ_qc, q, anc)

if f[0] == "constant":
    A = 1
else:
    A = 2

DJ_qc.h(anc[0])

print("After g: ")
oqf.Wavefunction(DJ_qc, systems=[3,A], show_systems=[True, False])


print("\nf type: ", f[0])
print("len(f) = ", len(f))
if len(f) > 1:
    print("States mapped to 1: ", f[1:len(f)])



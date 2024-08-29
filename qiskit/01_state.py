from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit_aer import Aer

q = QuantumRegister(1)
hello_qubit = QuantumCircuit(q)

hello_qubit.id(q[0])


S_simulator = Aer.backends(name="statevector_simulator")[0]
new_circ = transpile(hello_qubit, S_simulator)
S_simulator.run(new_circ)


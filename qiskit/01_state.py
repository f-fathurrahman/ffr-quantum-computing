from qiskit import QuantumRegister, QuantumCircuit, Aer, execute

q = QuantumRegister(1)
hello_qubit = QuantumCircuit(q)

hello_qubit.id(q[0])


S_simulator = Aer.backends(name="statevector_simulator")[0]

job = execute(hello_qubit, S_simulator)


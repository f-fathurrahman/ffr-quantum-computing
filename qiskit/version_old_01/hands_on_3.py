# https://obm.physics.metu.edu.tr/intro-QC/courses/PHYS437/Hands-On/hands-on-3/Hands-on-3-book.html

from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram

sim = Aer.get_backend("aer_simulator")
#backend = Aer.get_backend("statevector_simulator")

# Pauli gates
qc = QuantumCircuit(1)
qc.x(0) # apply PauliX to qubit-0
qc.draw() # let Qiskit decide the backend for plotting

# This is needed to avoid error: "No statevector for experiment"
qc.save_statevector() 
qobj = assemble(qc) # XXX vs transpile?
state = sim.run(qobj).result().get_statevector()
#plot_bloch_multivector(state)
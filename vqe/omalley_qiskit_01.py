from math import pi
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

theta = Parameter("theta") # only one parameter
qc = QuantumCircuit(2)
qc.x(0)
qc.barrier()
qc.rx(pi/2, 0) # negative sign (global phase)
qc.ry(pi/2, 1)
qc.cx(1, 0)
qc.rz(theta, 0)
qc.cx(1, 0)
qc.rx(pi/2, 0)
qc.ry(pi/2, 1)
qc.draw()


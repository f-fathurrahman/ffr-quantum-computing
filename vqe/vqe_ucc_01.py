# https://qiskit-community.github.io/qiskit-nature/howtos/vqe_ucc.html

from qiskit_nature.second_q.drivers import PySCFDriver
driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
problem = driver.run()

from qiskit_nature.second_q.mappers import JordanWignerMapper
mapper = JordanWignerMapper()

from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
ansatz = UCCSD(
    problem.num_spatial_orbitals,
    problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
    ),
)

import numpy as np
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit.primitives import Estimator # deprecated as of 1.2 ?
vqe = VQE(Estimator(), ansatz, SLSQP())

#from qiskit.primitives import StatevectorEstimator
#vqe = VQE(StatevectorEstimator(), ansatz, SLSQP())


# vqe.initial_point = np.zeros(ansatz.num_parameters)

# alternative
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint
initial_point = HFInitialPoint()
initial_point.ansatz = ansatz
initial_point.problem = problem
vqe.initial_point = initial_point.to_numpy_array()

from qiskit_nature.second_q.algorithms import GroundStateEigensolver
solver = GroundStateEigensolver(mapper, vqe)
result = solver.solve(problem)

print(f"Total ground state energy = {result.total_energies[0]:.4f}")
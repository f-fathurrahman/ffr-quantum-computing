import numpy as np
from qiskit.quantum_info import SparsePauliOp

A_op = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156),
])
print(f"Number of qubits: {A_op.num_qubits}")
print("A_op = ", A_op)

ref_value = np.linalg.eigvalsh(A_op.to_matrix())[0]
print(f"Reference value: {ref_value:.5f}")


# define ansatz and optimizer
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_algorithms.optimizers import SPSA

iterations = 125
#ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
ansatz = EfficientSU2(A_op.num_qubits)
spsa = SPSA(maxiter=iterations)
print("Number of parameters for ansatz = ", ansatz.num_parameters)


# without noise

# define callback
# note: Re-run this cell to restart lists before training
counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)



# define Aer Estimator for noiseless statevector simulation
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
seed = 1234
Nshots = 2**12
algorithm_globals.random_seed = seed
noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": Nshots},
    transpile_options={"seed_transpiler": seed},
)


# instantiate and run VQE
from qiskit_algorithms import VQE

vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result)
result = vqe.compute_minimum_eigenvalue(operator=A_op)

print(f"Nshots = {Nshots}")
print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}")

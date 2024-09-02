# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Introduction to Variational Quantum Eigensolver

# %% [markdown]
# In this tutorial we will learn an introduction to variational quantum eigensolver for calculation of lowest eigenvalue of a Hamiltonan or Hermitian matrix.

# %% [markdown]
# In VQE, a quantum computer and a classical computer work together to calculate the ground state energy of a molecule.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Let's review some stuffs first

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## What is Hermitian matrix ?

# %% [markdown]
# A Hermitian matrix is a square matrix $\mathbf{A}$ that satisfies:
# $$
# \mathbf{A} = \mathbf{A}^{\dagger}
# $$
# where $\mathbf{A}^{\dagger}$ is conjugate transpose of $\mathbf{A}$.
# For a real matrix, a Hermitian matrix is simply a symmetric matrix, i.e. a matrix which is equal to its transpose.

# %% [markdown]
# Hamiltonian matrix which is found in electronic structure problem (chemistry, material science, etc.) is an example of Hermitian matrix. 

# %% [markdown]
# Let's create a random matrix first.

# %%
A = np.random.rand(4,4)
A

# %% [markdown]
# ### Question
#
# Is matrix `A` Hermitian?

# %%
A.T - A

# %% [markdown]
# We can "make" `A` Hermitian or symmetric by taking only its lower triangule part or upper triangle part.
# Alternatively we can do something like this:

# %%
A = 0.5*(A.T + A)
A

# %%
A.T - A

# %% [markdown]
# ## Eigenvalue equation

# %% [markdown]
# An **eigenvalue** $\lambda$ of a matrix $\mathbf{A}$ obeys the following equation:
# $$
# \mathbf{A} \mathbf{x} = \lambda \mathbf{x}
# $$
# for some vector $\mathbf{x}$. The vector $\mathbf{x}$ is also called an **eigenvector**.

# %% [markdown]
# This eigenvalue equation appears in many applications. For quantum chemistry, eigenvalues are the energies of a system. We usually only concern ourselves with the lowest eigenvalue or the lowest energy.

# %%
λ = np.linalg.eigvalsh(A)

# %%
λ

# %%
λ, X = np.linalg.eigh(A)

# %%
λ[0]

# %%
x0 = X[:,0]

# %%
A @ x0

# %%
λ[0] * x0

# %%
np.dot(x0, x0)

# %% [markdown]
# ## Optimization

# %% [markdown]
# Optimizing a scalar function

# %%

# %% [markdown]
# # Qiskit

# %% [markdown]
# There are several things that we need to prepare before executing a VQE calculation.
#
# - The Hamiltonian matrix needs to be represented as tensor products of Pauli
# - Parameterized ansatz (recipe) for quantum circuit
# - minization algorithm

# %% [markdown]
# To create Pauli representation of a matrix, we can use `SparsePauliOp` class from Qiskit.

# %%
from qiskit.quantum_info.operators import SparsePauliOp

# %%
np.random.seed(1234)
A = np.random.rand(4,4) + 10*np.eye(4) # create a digonally dominant matrix
A = 0.5*(A + A.T)
A

# %%
pauli_op = SparsePauliOp.from_operator(A)

# %%
pauli_op

# %% [markdown]
# ## Checking Pauli representation

# %%
pauli_op_list = pauli_op.to_list()
pauli_op_list

# %%
import gate_matrix


# %%
def process_gate_str(g_str):
    Nqubit = len(g_str)
    matRes = np.eye(Nqubit, dtype=np.complex128)
    mat_list = []
    for s in g_str:
        if s == 'X':
            mat_list.append(gate_matrix.X)
        elif s == 'Y':
            mat_list.append(gate_matrix.Y)
        elif s == 'Z':
            mat_list.append(gate_matrix.Z)
        elif s == 'I':
            mat_list.append(gate_matrix.I)
        else:
            raise RuntimeError(f"Unknown gate: {s}")
    Nmat = len(mat_list)
    if Nmat >= 2:
        A = np.kron(mat_list[0], mat_list[1])
        for i in range(2,Nmat):
            A = np.kron(A, mat_list[2])
    else:
        A = mat_list[0]
    return A


# %%
matA = np.zeros( (4,4), dtype=np.complex128)
for p in pauli_op_list:
    matA += p[1] * process_gate_str(p[0])

# %%
np.real(matA)

# %% [markdown]
# Original matrix:

# %%
A

# %% [markdown]
# ## Qiskit: built in algorithm

# %%
ref_value = np.linalg.eigvalsh(pauli_op.to_matrix())[0]
print(f"Reference value: {ref_value:.5f}")

# %%
# define ansatz and optimizer
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms.optimizers import SPSA

# %%
import qiskit.circuit.library
import qiskit_algorithms.optimizers

# %%
iterations = 125
ansatz = EfficientSU2(pauli_op.num_qubits)
spsa = SPSA(maxiter=iterations)

# %%
# define callback
# note: Re-run this cell to restart lists before training
counts = []
values = []
def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


# %%
from qiskit_algorithms.utils import algorithm_globals
seed = 1234
algorithm_globals.random_seed = seed

# %%
Nshots = 2**10
# define Aer Estimator for noiseless statevector simulation
from qiskit_aer.primitives import Estimator as AerEstimator
noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": Nshots},
    transpile_options={"seed_transpiler": seed},
)

# %%
# instantiate and run VQE
from qiskit_algorithms import VQE

# %%
vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result)

# %%
result = vqe.compute_minimum_eigenvalue(operator=pauli_op)

print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}")

# %%
import matplotlib.pyplot as plt

# %%
plt.style.use("dark_background")

# %%
plt.plot(counts, values);

# %%

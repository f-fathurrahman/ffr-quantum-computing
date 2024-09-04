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

# %%
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use("dark_background") # disable this for default style

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# %%
from math import pi

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Statevector

# %% [markdown]
# One qubit state can be represented using statevector:
# $$
# \left| \psi \right\rangle = \begin{bmatrix}
# a \\
# b
# \end{bmatrix}
# $$
# where $a$ and $b$ are complex numbers.

# %% [markdown]
# State vector is normalized according to:
# $$
# \left \langle \psi | \psi \right \rangle = 1
# $$
# with
# $$
# \left \langle \psi \right| = \begin{bmatrix}
# a^{*} & b^{*}
# \end{bmatrix}
# $$
# where $a^{*}$ and $b^{*}$ are complex conjugate of $a$ and $b$, respectively.

# %% [markdown]
# Let's create a random state vector, this is our $| \psi \rangle$

# %%
np.random.seed(1234)
psi = np.random.rand(2,1) + np.random.rand(2,1)*1j
psi

# %% [markdown]
# And this is our $\langle \psi |$

# %%
psi.conj().T

# %% [markdown]
# The norm of $\psi$ is:

# %%
np.sqrt( psi.conj().T @ psi )

# %% [markdown]
# Is this statevector normalized?

# %% [markdown]
# If not, how to normalize this statevector?

# %%
c = np.sqrt( psi.conj().T @ psi ) # c is actually a 1x1 matrix
psi *= (1/c)

# %%
psi.conj().T @ psi

# %% [markdown]
# An alternative way to compute norm (this will return a scalar):

# %%
np.linalg.norm(psi)

# %% [markdown]
# ### Plotting statevector (Optional)

# %% [markdown]
# One qubit statevector can be visualized as a point on sphere. To do this we must convert statevector elements $\alpha$ and $\beta$ to spherical coordinate with radial coordinate equal to 1 (in case statevector is normalized).

# %%
from qiskit.visualization import plot_bloch_vector

# %%
# Modified from:
# https://quantumcomputing.stackexchange.com/questions/10116/how-to-get-the-bloch-sphere-angles-given-an-arbitrary-qubit
import cmath
import math
def qubit_to_spherical(psi):
    Ndim = len(psi.shape)
    if Ndim == 2:
        α = psi[0,0]
        β = psi[1,0]
    elif Ndim == 1:
        α = psi[0]
        β = psi[1]
    r = math.sqrt(abs(α)**2 + abs(β)**2)
    α /= r # normalize
    β /= r # normalize
    θ = cmath.acos(abs(α))*2
    if θ:
        ϕ = cmath.log(β * cmath.exp(-1j*cmath.phase(α)) / cmath.sin(θ/2)) / 1j
    else: # ?
        ϕ = 0.0
    return θ.real, ϕ.real


# %%
qubit_to_spherical(np.array([0.0, 1.0]))

# %%
plot_bloch_vector([1,pi/3,0.1], coord_type="spherical")

# %% [markdown]
# ### Using Statevector

# %% [markdown]
# An easier way is to convert Numpy array to Qiskit `Statevector` object and then use `plot_bloch_multivector`.

# %%
from qiskit.quantum_info.states import Statevector

# %%
psi_v = Statevector(psi)

# %%
psi_v

# %%
from qiskit.visualization import plot_bloch_multivector

# %%
plot_bloch_multivector(psi_v)

# %% [markdown]
# ## Computational basis vector

# %% [markdown]
# Statevector can be written in terms of basis vectors. This is similar to how we can decompose a point coordinate into $x$ and $y$ (in plane). Because Hilbert space size of a qubit is two, we need two basis vectors.
#
# There are several many basis vectors that can be used. One important basis vector is "computational" basis.

# %% [markdown]
# Computational basis is defined as:
# $$
# \left| 0 \right\rangle = \begin{bmatrix}
# 1 \\ 0
# \end{bmatrix}
# $$
#
# $$
# \left| 1 \right\rangle = \begin{bmatrix}
# 0 \\ 1
# \end{bmatrix}
# $$
#
# Note that 0 and 1 within $\ket{0}$ and $\ket{1}$ represent symbols, not a number.

# %%
ket0 = np.array([ [1], [0] ])
ket1 = np.array([ [0], [1] ])

# %% [markdown]
# `ket0` and `ket1` should be orthonormal: 

# %%
ket0.conj().T @ ket0

# %%
ket0.conj().T @ ket1

# %% [markdown]
# Using computational basis, we can write $\left| \psi \right\rangle$ linear combination of this basis set:
# $$
# \psi = c_0 \left| 0 \right\rangle + c_1 \left| 1 \right\rangle
# $$
#
# How do we find the coefficients $c_0$ and $c_1$ ?

# %%
c0 = ket0.conj().T @ psi
c1 = ket1.conj().T @ psi

# %% [markdown]
# Let's evaluate the expansion using c0 and c1 along with basis set:

# %%
c0*ket0 + c1*ket1

# %% [markdown]
# They should be the same with the original statevector:

# %%
psi

# %% [markdown]
# ## Other bases

# %% [markdown]
# $$
# \left| + \right\rangle = \begin{bmatrix}
# \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}}
# \end{bmatrix}
# $$
#
# $$
# \left| - \right\rangle = \begin{bmatrix}
# \frac{1}{\sqrt{2}} \\ -\frac{1}{\sqrt{2}}
# \end{bmatrix}
# $$

# %%
ketPlus = np.array( [ [1/np.sqrt(2)], [1/np.sqrt(2)] ])
ketMinus = np.array( [ [1/np.sqrt(2)], [-1/np.sqrt(2)] ])

# %%
ketPlus.conj().T @ ketPlus

# %%
ketPlus.conj().T @ ketMinus

# %% [markdown]
# $$
# \left| +\imath \right\rangle = \begin{bmatrix}
# \frac{1}{\sqrt{2}} \\ \frac{\imath}{\sqrt{2}}
# \end{bmatrix}
# $$
#
# $$
# \left| -\imath \right\rangle = \begin{bmatrix}
# \frac{1}{\sqrt{2}} \\ -\frac{\imath}{\sqrt{2}}
# \end{bmatrix}
# $$

# %%
ketPlusI = np.array([ [1/np.sqrt(2)], [1j/np.sqrt(2)] ])
ketMinusI = np.array([ [1/np.sqrt(2)], [-1j/np.sqrt(2)] ])

# %%
ketPlusI.conj().T @ ketPlusI 

# %%
ketPlusI.conj().T @ ketMinusI

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Two qubits

# %% [markdown]
# Statevector of two qubit can be written as:
# $$
# \left| \psi \right\rangle = \begin{bmatrix}
# a \\ b \\ c \\ d
# \end{bmatrix}
# $$
# where $a$, $b$, $c$, and $d$ are complex numbers.

# %% [markdown]
# They also can be written as basis set expansion:
# $$
# \ket{\psi} = c_{00} \ket{00} + c_{01} \ket{01} + c_{10} \ket{10} + c_{11} \ket{11}
# $$

# %% [markdown]
# The basis vectors are tensor products of one qubit basis vectors.

# %% [markdown]
# We will be using the following convention about tensor products.
#
# Given one qubit in with statevector $\ket{q_0}$ and another qubit in state $\ket{q_1}$, the tensor product $\ket{q_0}$ and $\ket{q_1}$ is
# $$
# \ket{q_1 q_0} = \ket{q_1} \otimes \ket{q_0}
# $$
# where $\otimes$ is tensor product or Kronecker product operator.

# %% [markdown]
# $$
# \begin{eqnarray}
# \ket{00} &= \ket{0} \otimes \ket{0} \\
# \ket{01} &= \ket{0} \otimes \ket{1} \\
# \ket{10} &= \ket{1} \otimes \ket{0} \\
# \ket{11} &= \ket{1} \otimes \ket{1}
# \end{eqnarray}
# $$

# %% [markdown]
# In Numpy, we can use `np.kron` to compute tensor product.

# %%
ket00 = np.kron(ket0, ket0)
ket00

# %%
ket01 = np.kron(ket0, ket1)
ket01

# %%
ket10 = np.kron(ket1, ket0)
ket10

# %%
ket11 = np.kron(ket1, ket1)
ket11

# %% [markdown]
# ### Your task
#
# Check orthonormality of `ket00`, `ket01`, `ket10`, and `ket11`

# %% [markdown]
# ### Your task
#
# Create a random complex (state)vector with 4 elements and normalize it.
# Then compute the expansion coefficients $c_{00}$, $c_{01}$, etc. Verify that the expansion is the same as original vector.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Three, four, ... qubits

# %% [markdown]
# For three qubits we can represent statevector with column vector with $2^3$ elements.
# In general for $N$ qubits we have statevector with $2^{N}$ elements.

# %% [markdown]
# Following the convention we mentioned before, for three qubits, we have tensor products:
# $$
# \ket{q_2 q_1 q_0} = \ket{q_2} \otimes \ket{q_1} \otimes \ket{q_0}
# $$

# %% [markdown]
# The basis vectors also can be computed using tensor products.
# For example:
# $$
# \ket{001} = \ket{0} \otimes \ket{0} \otimes \ket{1}
# $$

# %%
ket001 = np.kron( np.kron(ket0, ket0), ket1)
ket001

# %%
ket100 = np.kron( np.kron(ket1, ket0), ket0)
ket100

# %% [markdown]
# ## Operator, quantum gate

# %% [markdown]
# An operator $\hat{O}$ map a statevector to another statevector. We can write:
# $$
# \ket{\psi'} = \hat{O} \ket{\psi}
# $$

# %% [markdown]
# In quantum computing, we process or evolve the statevector by using various operators. These operators will be called as quantum gates.

# %% [markdown]
# We usually represent an operator as a matrix. Dimension of the matrix depends on the dimension of the statevector it operates on. For example, if an operator or quantum gate operate only on one qubit, then the dimension of the matrix is $2 \times 2$.
# For two qubits then the matrix dimension is $4 \times 4$, etc.

# %% [markdown]
# Suppose that we have two qubits with statevector
# $$
# \ket{\psi} = \ket{q_1 q_0} = \ket{q_1} \otimes \ket{q_0}
# $$

# %% [markdown]
# $\hat{A}$ is an operator that operates on qubit $q_1$ and $\hat{B}$ is an operator that operates on $q_0$.
# The actions of these operators can be written as:
# $$
# \left( \hat{A} \ket{q_1} \right) \otimes \left( \hat{B} \ket{q_0} \right) 
# $$

# %% [markdown]
# Note that, both matrix representations of $\hat{A}$ and $\hat{B}$ has size of $2\times2$ because each of them operates on one qubit.

# %% [markdown]
# These operations also can be written as:
# $$
# \hat{C} \ket{q_1 q_0}
# $$
# where matrix representation of $\hat{C}$ must have size of $4 \times 4$. This matrix can be written as:
# $$
# \hat{C} = \hat{A} \otimes \hat{B}
# $$

# %% [markdown]
# ### Unitary operator

# %% [markdown]
# If both $\ket{\psi}$ and $\ket{\psi'}$ are normalized, the operator $\hat{O}$ is said to be unitary. A unitary matrix $U$ satisfies the following relation:
# $$
# U U^{\dagger} = U^{\dagger} U = I
# $$
# where $I$ is an identity matrix and ${U}^{\dagger}$ is conjugate transpose of $U$.

# %% [markdown]
# Most of quantum gates that we will use are unitary. They will not change the norm of the statevector.

# %% [markdown]
# ### Expectation value

# %% [markdown]
# For a given operator $\hat{O}$ we can define expectation value of that operator with respect to statevector $\ket{\psi}$:
# $$
# \braket{O} = \braket{\psi | \hat{O} | \psi}
# $$

# %% [markdown]
# ### Hermitian operator

# %% [markdown]
# In quantum mechanics, any observable or measurable quantities must be represented by Hermitian operators or matrices. A Hermitian matrix $\mathcal{H}$ satisfies:
# $$
# \mathcal{H} = \mathcal{H}^{\dagger}
# $$
# Expectation value of Hermitian matrix is a real number.

# %% [markdown]
# ## One qubit quantum gates

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Identity

# %% [markdown]
# $$
# I = \begin{bmatrix} 
# 1 & 0 \\
# 0 & 1 
# \end{bmatrix}
# $$

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Pauli matrices

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# $$
# {X} = \begin{bmatrix} 
# 0 & 1 \\
# 1 & 0 
# \end{bmatrix}\qquad
# {Y} = \begin{bmatrix} 
# 0 & -i \\
# i & 0 
# \end{bmatrix}\qquad
# {Z} = \begin{bmatrix} 
# 1 & 0 \\
# 0 & -1 
# \end{bmatrix}
# $$

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Hadamard matrix

# %% [markdown]
# $$
# {H} = \frac{1}{\sqrt{2}}\begin{bmatrix} 
# 1 & 1 \\
# 1 & -1 
# \end{bmatrix}
# $$

# %% [markdown]
# ### Rotation matrices

# %% [markdown]
# These matrices have by one parameter $\theta$.

# %% [markdown]
# $$
# R_x(\theta) = \begin{bmatrix} 
# \cos(\theta/2)     & -\imath \sin(\theta/2) \\
# -\imath \sin (\theta/2) & \cos(\theta/2) 
# \end{bmatrix}
# $$
# $$
# R_y(\theta) = \begin{bmatrix} 
# \mathrm{cos}(\theta/2) & -\mathrm{sin}(\theta/2) \\
# \mathrm{sin}(\theta/2) & \mathrm{cos}(\theta/2) 
# \end{bmatrix}
# $$
# $$
# R_z(\theta) = \begin{bmatrix} 
# \mathrm{exp}(-i\theta/2) & 0 \\
# 0                        & \mathrm{exp}(i\theta/2) 
# \end{bmatrix}
# $$

# %% [markdown]
# ### Qiskit quantum gates

# %% [markdown]
# We can code these matrices manually using Numpy. Alternatively, we can use Qiskit circuit library to obtain matrix representations of of quantum gates.

# %%
import qiskit.circuit.library as qclib

# %%
I = qclib.IGate().to_matrix()

# %%
X = qclib.XGate().to_matrix()
Y = qclib.YGate().to_matrix()
Z = qclib.ZGate().to_matrix()

# %%
H = qclib.HGate().to_matrix()

# %%
RX = qclib.RXGate(0.1).to_matrix() # chose theta=0.1

# %% [markdown]
# ### Your tasks

# %% [markdown]
# #### Verify that these matrices are unitary.

# %% [markdown]
# #### Try to apply these matrices basis set, for example try to apply $X$ to $\ket{0}$ and $\ket{1}$ and visualize the results.

# %% [markdown]
# #### Compute expectation values of various basis set vector for Pauli matrices.

# %%
psi = ket0;
res = psi.conj().T @ (Z @ psi)
res

# %%
psi = ketMinus;
res = psi.conj().T @ (X @ psi)
res

# %%
psi = ketPlusI
res = psi.conj().T @ (Z @ psi)
res

# %% [markdown]
# ## Quantum circuit

# %% [markdown]
# In Qiskit, we don't usually work with matrix representation of quantum gates. Instead we build a quantum circuit and add the required gates or operations.

# %%
from qiskit import QuantumCircuit

# %% [markdown]
# In the following example, we build a QuantumCircuit with 2 qubits $\ket{q_1 q_0}$, apply Hadamard gate to $q_1$ and Pauli $X$ gate the $q_0$, the we measure the results. We also draw the corresponding circuit.

# %%
qc = QuantumCircuit(2)
qc.h(1) # Hadamard gate to q1
qc.x(0) # Hdamard gate to q0
qc.measure_all()
qc.draw()

# %% [markdown]
# Using Matplotlib

# %%
qc.draw("mpl")

# %% [markdown]
# ### Quantum circuit simulation

# %% [markdown]
# In Qiskit, we can simulate the operations of the quantum circuit on classical computers. We usually use two kind of simulators:
# - Statevector simulator: this is similar to using raw matrix multiplications
# - QASM simulator: this is a probabilistic simulator, to simulate real measurement. We usually use this for quantum circuits which have measurement gates.

# %%
from qiskit import transpile
from qiskit_aer import Aer

def run_on_qasm_simulator(my_qc, shots=1000):
    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc, shots=shots)
    return job.result()

def run_on_statevector_simulator(my_qc):
    backend = Aer.get_backend("statevector_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc)
    return job.result()


# %% [markdown]
# ### Example simulations

# %%
qc = QuantumCircuit(1)
qc.h(0)
qc.draw()

# %%
res = run_on_statevector_simulator(qc)
res.get_statevector()

# %%
res.get_statevector().data.real

# %%
1/np.sqrt(2) * (ket0 + ket1)

# %%
qc = QuantumCircuit(1)
qc.h(0)
qc.measure_all()
qc.draw()

# %%
res = run_on_qasm_simulator(qc)
res.get_counts()

# %% [markdown]
# What happen if we run statevector simulator on this quantum circuit?

# %%
res = run_on_statevector_simulator(qc)
res.get_statevector()

# %% [markdown]
# ### Multiple qubits quantum gates

# %% [markdown]
# We can build quantum gates that operates on multiple qubits by using tensor products of single qubit quantum gates.

# %% [markdown]
# Usually multiple qubit quantum gates are controlled gates.
# Controlled gates need more than one qubits to operate.
# They are somewhat equivalent to conditional statements in the classical programming languages. The examples are controlled X gate (CX), controlled-Y gate (CY), etc.
# CX gate need two qubits: first qubit is the control qubit, second qubit is the target qubit.

# %%
qc = QuantumCircuit(2)
qc.x(0)
qc.cx(0,1) # q0 is the control, q1 is target
qc.measure_all()
qc.draw()

# %%
res = run_on_qasm_simulator(qc)
res.get_counts()

# %% [markdown]
# ### Parametrized quantum circuit

# %% [markdown]
# Some gates need parameter(s), such as RX gate.

# %%
qc = QuantumCircuit(1)
qc.rx(0.1, 0) # 0.1 is theta
qc.draw()

# %%
from qiskit.circuit import ParameterVector

# %%
theta = ParameterVector("theta", 1)

# %%
qc = QuantumCircuit(1)
qc.rx(theta[0], 0)
qc.draw()

# %%
# res = run_on_statevector_simulator(qc) # this will error

# %%
tqc = qc.assign_parameters([pi])
res = run_on_statevector_simulator(tqc)
res.get_statevector()

# %%
from qiskit.visualization import plot_bloch_multivector
plot_bloch_multivector(res.get_statevector())

# %% [markdown]
# ## Last cell

# %%

# %% [markdown]
# ## 

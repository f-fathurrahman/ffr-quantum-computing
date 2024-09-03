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
matplotlib.style.use("dark_background") # disable this for default style

# %%
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# %% [markdown]
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
psi.conj().T @ psi

# %% [markdown]
# Is this statevector normalized?

# %% [markdown]
# If not, how to normalize this statevector?

# %%
c = psi.conj().T @ psi # this is norm squared
psi *= (1/np.sqrt(c))

# %%
psi.conj().T @ psi

# %%
np.linalg.norm(psi)

# %% [markdown]
# ### Plotting statevector (Optional)

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
plot_bloch_vector([1,math,pi,0], coord_type="spherical")

# %%

# %%
plot_bloch_vector([1,theta,phi], coord_type=

# %% [markdown]
# ## Computational basis vector

# %% [markdown]
# Statevector can be written in terms of basis vectors. There are several many basis vectors that can be used.

# %% [markdown]
# Computational basis:
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

# %%

# %%

# %%

# %% [markdown]
# ## Last cell

# %% [markdown]
# ## 

# %% [markdown]
# ## Operator multikubit

# %% [markdown]
# Gerbang Y2, operasi pada kubit ke-2 (dihitung dari kiri, kubit ke-0 adalah kubit paling kiri):
# ```python
# Y2 = ops.Identity() * ops.Identity() * ops.PauliY
# ```
#
# Dalam notasi matematika:
# $$
# Y_2 = I \otimes I \otimes Y = I^{\otimes 2} \otimes Y
# $$

# %% [markdown]
# Penggunaaan istilah berikut kurang lebih sama: gerbang (*gate*), operator, matriks
#
# $$
# \ket{\psi} = \alpha \ket{0} + \beta \ket{1}
# $$
#
# $$
# \ket{\phi} = \hat{O} \ket{\psi} = \alpha' \ket{0} + \beta' \ket{1}
# $$

# %% [markdown]
# ## Gerbang identitas

# %% [markdown]
# Dalam bentuk matriks:
#
# $$
# I = \begin{bmatrix}
# 1 & 0 \\
# 0 & 1
# \end{bmatrix}
# $$
#
# $$
# \begin{bmatrix}
# 1 & 0 \\
# 0 & 1
# \end{bmatrix}
# \begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
# \begin{bmatrix} \alpha \\ \beta \end{bmatrix}
# $$

# %%
from qc4p import ops, state

I = ops.Identity()

I.dump()

# %% [markdown]
# ## Gerbang Pauli

# %% [markdown]
# Notasi:
#
# - $\sigma_{x}$, $\sigma_{y}$, $\sigma_z$ atau
#
# - $\sigma_{1}$, $\sigma_{2}$, $\sigma_3$
#
# Gerbang Pauli X dikenal juga sebagai gerbang X, gerbang kuantum NOT atau X.
#
# Bentuk matriks:
# $$
# X = \begin{bmatrix}
# 0 & 1 \\
# 1 & 0
# \end{bmatrix}
# $$
#
# Aksi gerbang ini pada keadaan basis:
# $$
# \begin{align}
# X \ket{0} & = \ket{1} \\
# X \ket{1} & = \ket{0}
# \end{align}
# $$
#
# Gerbang Pauli X menukar koefisien $\alpha$ dan $\beta$ dari suatu kubit.
# $$
# X \ket{\psi} = \begin{bmatrix}
# 0 & 1 \\
# 1 & 0
# \end{bmatrix}
# \begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
# \begin{bmatrix} \beta \\ \alpha \end{bmatrix}
# $$
#
#
# Gerbang Pauli Y
# $$
# Y \ket{\psi} = \begin{bmatrix}
# 0 & -\imath \\
# \imath & 0
# \end{bmatrix}
# \begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
# \begin{bmatrix} -\imath \beta \\ \imath \alpha \end{bmatrix}
# $$
#
#
# Gerbang Pauli Z (phase flip gate), membalikkan tanda dari komponen kubit kedua
# (tanda dari koefisien $\beta$):
# $$
# Z \ket{\psi} = \begin{bmatrix}
# 1 & 0 \\
# 0 & -1
# \end{bmatrix}
# \begin{bmatrix} \alpha \\ \beta \end{bmatrix} =
# \begin{bmatrix} \alpha \\ -\beta \end{bmatrix}
# $$
#
#
# Sifat gerbang Pauli (involutory):
# $$
# XX = YY = ZZ = I
# $$

# %%
X = ops.PauliX()
X.dump()

# %%
XX = X(X)

XX.dump()

# %%
psi = state.qubit(beta=1.0)
psi.dump()

import numpy as np

np.sqrt(0.75)

# %% [markdown]
# Aplikasikan operator Pauli X

# %%
Y = ops.PauliY()

print("Original: ")
psi = state.qubit(beta=-0.5)
psi.dump()

# %%
print()
print("After application of Y gate: ")
phi = Y(psi)
phi.dump()

# %%
Z = ops.PauliZ()

print("Original: ")
psi = state.qubit(beta=0.5)
psi.dump()

# %%
print()
print("After application of Z gate: ")
phi = Z(psi)
phi.dump()

# %% [markdown]
# ## Visualisasi

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("dark_background")

# %%
from qiskit.visualization import plot_bloch_vector
from qc4p import helper

# %%
plot_bloch_vector(helper.qubit_to_bloch(psi))

# %%
plot_bloch_vector(helper.qubit_to_bloch(phi))

# %% [markdown]
# ## Gerbang Rotasi

# %% [markdown]
# Gerbang rotasi dikonstruksi menggunakan eksponensiasi dari matriks atau gerbang Pauli.
#
# $$
# \ket{\psi} = \cos\left( \frac{\theta}{2} \right) \ket{0} +
# e^{\imath \phi} \sin\left( \frac{\theta}{2} \right) \ket{1}
# $$
#
# Rotasi disekitar sumbu-x, y, dan z
# $$
# \begin{align}
# R_{x}(\theta) & = \exp\left[ -\imath \frac{\theta}{2} X \right] \\
# R_{y}(\theta) & = \exp\left[ -\imath \frac{\theta}{2} Y \right] \\
# R_{z}(\theta) & = \exp\left[ -\imath \frac{\theta}{2} Z \right]
# \end{align}
# $$
#
# Untuk matriks involutori $A$:
# $$
# \exp\left[ \imath \theta A \right] = \cos(\theta) I + \imath \sin(\theta) A
# $$

# %% [markdown]
# $$
# \begin{align}
# R_{x}(\theta) & = \exp\left[ -\imath \frac{\theta}{2} X \right] \\
# & = \cos(\theta) I - \imath \sin(\theta) X \\
# & = \begin{bmatrix}
# \cos(\frac{\theta}{2}) & -\imath \sin(\frac{\theta}{2}) \\
# -\sin(\frac{\theta}{2}) & \cos(\frac{\theta}{2}) \\
# \end{bmatrix}
# \end{align}
# $$

# %%
θ = np.deg2rad(45)
Rx45 = ops.RotationX(θ)
Rx45.dump()

# %%
print("Original: ")
psi = state.qubit(beta=0.5)
psi.dump()

print()
print("After application of Rx45 gate: ")
phi = Rx45(psi)
phi.dump()

# %%
plot_bloch_vector(helper.qubit_to_bloch(psi))

# %%
plot_bloch_vector(helper.qubit_to_bloch(phi))

# %% [markdown]
# ## Gerbang Fasa

# %% [markdown]
# Atau gerbang S atau gerbang Z90: merepresentasikan fasa 90 derajat di sekitar sumbu-z untuk bagian
# $\ket{1}$ dari kubit. 
#
# $$
# S = \begin{bmatrix}
# 1 & 0 \\
# 0 & \imath
# \end{bmatrix}
# $$

# %% [markdown]
# Gerbang S hanya mempengaruhi komponen kedua dari kubit dengan pengali $\imath$
# (merepresentasikan rotasi 90$^\circ$)
#
# Gerbang Rz secara umum mempengaruhi kedua komponen dari kubit.
#
# Hubungan antara gerbang S dan gerbang Z:
# $$
# S^2 = Z
# $$

# %% [markdown]
# ## Gerbang fasa fleksibel

# %% [markdown]
# Gerbang fasa diskrit, disimbolkan dengan $R_{k}$ atau gerbang `Rk`, merupakan
# generalisasi dari gerbang fasa, yang melakukan rotasi di sekitar sumbu-z dengan
# pangkat pecahan dari 2, yaitu $2\pi/2^k$, untuk k > 0.
# Contohnya: $\pi$, $\pi/2$, $\pi/4$, $\pi/8$, dan seterusnya.
#
# $$
# R_{k}(k) = \begin{bmatrix}
# 1 & 0 \\
# 0 & e^{2\pi/2^k}
# \end{bmatrix}
# $$
#

# %% [markdown]
# Bentuk yang lain adalah gerbang `U1(lambda)` yang juga dikenal sebagai gerbang pergeseran
# fasa atau *phase kick gate*:
# $$
# U_{1}(\lambda) = \begin{bmatrix}
# 1 & 0 \\
# 0 & e^{\imath \lambda}
# \end{bmatrix}
# $$
#
# Gerbang ini sama dengan $R_{k}$, hanya saja sembarang sudut fasa diperbolehkan.
#
# Beberapa gerbang-gerbang bernama yang lain merupakan kasus khusus dari gerbang
# $R_{k}$.
# $$
# \begin{align}
# R_{k}(0) & = I \\
# R_{k}(1) & = Z \\
# R_{k}(2) & = S \\
# R_{k}(3) & = T
# \end{align}
# $$

# %% [markdown]
# ## Gerbang Akar Kuadrat

# %% [markdown]
# Akar kuadrat dari gerbang X adalah gerbang V.
# Gerbang $V$ bersifat uniter $VV^{\dagger} = I$ namun juga
# memenuhi $V^2 = X$.
# Gerbang ini didefinisikan sebagai:
# $$
# V = \frac{1}{2} \begin{bmatrix}
# 1 + \imath & 1 - \imath \\
# 1 - \imath & 1 + \imath
# \end{bmatrix}
# $$
#

# %% [markdown]
# Akar dari gerbang fasa dikenal dengan gerbang-T.
# Gerbang S merepresentasikan fasa $90^{\circ}$ disekitar sumbu-z. Gerbang
# $T$ merepresentasikan fasa $45^{\circ}$.
# $$
# T = \begin{bmatrix}
# 1 & 0 \\
# 0 & e^{\imath \pi/4}
# \end{bmatrix}
# $$

# %% [markdown]
# Akar dari gerbang Y:
# $$
# Y_{\mathrm{root}} = \frac{1}{2}\begin{bmatrix}
# 1 + \imath & -1 - \imath \\
# 1 + \imath &  1 + \imath
# \end{bmatrix}
# $$

# %% [markdown]
# Untuk bentuk matriks atau operator umum, kita dapat menggunakan fungsi
# `sqrtm` dari modul `scipy.linalg`.
#

# %% [markdown]
# ## Operator proyeksi

# %% [markdown]
# Operator proyeksi untuk suatu keadaan adalah hasil kali luar
# (*outer product*) dari keadaan tersebut dengan dirinya sendiri.
# Misalnya, operator proyeksi untuk keadaan $\ket{0}$ adalah
# $$
# P_{\ket{0}} = \ket{0}\bra{0} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
# \begin{bmatrix} 1 & 0 \end{bmatrix} = 
# \begin{bmatrix}
# 1 & 0 \\
# 0 & 0
# \end{bmatrix}
# $$
#
# Untuk $\ket{1}$ adalah:
# $$
# P_{\ket{1}} = \ket{1}\bra{1} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
# \begin{bmatrix} 0 & 1 \end{bmatrix} = 
# \begin{bmatrix}
# 0 & 0 \\
# 0 & 1
# \end{bmatrix}
# $$

# %% [markdown]
# Operator proyeksi buka gerbang kuantum, karena tidak bersifat uniter
# dan tidak reversibel. Meskipun demikian operator ini tetap digolongkan sebagai
# operator kuantum karena bersifat Hermitian: $P = P^{\dagger}$.
# Untuk keadaan ternormalisasi, operator proyeksi bersifat idempoten:
# $P = P^2$


# %% [markdown]
# ## Gerbang Hadamard

# %% [markdown]
# Dalam bentuk matriks:
# $$
# H = \frac{1}{\sqrt{2}}
# \begin{bmatrix}
# 1 & 1 \\
# 1 & -1
# \end{bmatrix}
# $$
#
#
# Aksi gerbang Hadamard pada $\ket{0}$
# $$
# H\ket{0} = \frac{\ket{0} + \ket{1}}{\sqrt{2}} = \ket{+}
# $$
#
# Aksi gerbang Hadamard pada $\ket{1}$
# $$
# H\ket{1} = \frac{\ket{0} - \ket{1}}{\sqrt{2}} = \ket{-}
# $$
#
# Gerbang Hadamard membuat mengubah suatu kubit menjadi superposisi dari
# dari dua keadaan basis.
#
# Untuk keadaan $\ket{\psi} = \alpha\ket{0} + \beta\ket{1}$:
# $$
# \begin{align}
# H \ket{\psi} & = \alpha H \ket{0} + \beta H \ket{1} \\
# & = \alpha \ket{+} + \beta \ket{-} \\
# & = \frac{\alpha + \beta}{\sqrt{2}} \ket{0} +
#     \frac{\alpha - \beta}{\sqrt{2}} \ket{1}
# \end{align}
# $$
#
# Gerbang Hadamard bersifat involutori.
# $$
# HH = I
# $$

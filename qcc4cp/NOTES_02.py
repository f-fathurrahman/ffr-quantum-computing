# %% [markdown]
# # Gerbang Kubit Terkontrol

# %% [markdown]
r"""
Tinjau contoh gerbang CNOT (Controlled NOT) atau CX yang memiliki representasi matriks
(dalam basis komputasi $\ket{0}$ dan $\ket{1}$)
$$
\mathrm{CX}_{0,1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$
"""


# %% [markdown]
# Aplikasi matriks ini pada keadaan $\ket{00}$ dan $\ket{01}$ tidak mengubah dua keadaan tersebut.

# %%
import matplotlib.pyplot as plt


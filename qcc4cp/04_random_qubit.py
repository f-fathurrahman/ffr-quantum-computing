import scipy
import numpy as np
from qc4p import state
from qc4p import helper

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "savefig.dpi": 150
})

from qiskit.visualization import plot_bloch_vector

Umat = scipy.stats.unitary_group.rvs(2)
eigvals, eigvecs = np.linalg.eig(Umat)
psi = state.State(eigvecs[:,0])

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(psi))
plt.savefig("IMG_random_q.png")


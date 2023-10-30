import numpy as np
from qc4p import state
from qc4p import helper

import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "savefig.dpi": 150
})

from qiskit.visualization import plot_bloch_vector

ket0 = state.zeros(1)
ket1 = state.ones(1)

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ket0))
plt.savefig("IMG_ket0.png")

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ket1))
plt.savefig("IMG_ket1.png")


# Hadamard basis
ketPlus = state.plus()
ketMinus = state.minus()

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ketPlus))
plt.savefig("IMG_ketPlus.png")

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ketMinus))
plt.savefig("IMG_ketMinus.png")


# ??? the name?
ketPlusI = state.plusi()
ketMinusI = state.minusi()

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ketPlusI))
plt.savefig("IMG_PlusI.png")

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(ketMinusI))
plt.savefig("IMG_MinusI.png")


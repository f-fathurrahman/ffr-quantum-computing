import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("dark_background")

from qc4p import state, ops, helper

from qiskit.visualization import plot_bloch_vector

psi = state.qubit(alpha=0.5)

plt.clf()
plot_bloch_vector(helper.qubit_to_bloch(psi))
plt.savefig("IMG_0.png", dpi=150)

Δθ = 20
Rx = ops.RotationZ(np.deg2rad(Δθ))
for i in range(1,19):
    psi = Rx(psi)
    plt.close()
    psi.dump()
    c_ball = helper.qubit_to_bloch(psi)
    plot_bloch_vector(c_ball)
    plt.savefig("IMG_" + str(i) + ".png", dpi=150)
    print("θ = ", Δθ*i, " c_ball = ", c_ball, " magn = ", np.linalg.norm(np.array(c_ball)))


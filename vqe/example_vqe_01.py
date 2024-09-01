import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

from scipy.optimize import minimize

import matplotlib.pyplot as plt

from qiskit_aer import Aer
backend = Aer.get_backend("qasm_simulator")

# Map classical inputs to a quantum problem
hamiltonian = SparsePauliOp.from_list([
    ("YZ", 0.3980),
    ("ZI", -0.3980),
    ("ZZ", -0.0133),
    ("XX", 0.1810)
])

# The ansatz is EfficientSU2
ansatz = EfficientSU2(hamiltonian.num_qubits)
#ansatz.decompose().draw("mpl", style="iqp")
#plt.savefig("IMG_ansatz_efficient_SU2.pdf")
num_params = ansatz.num_parameters
print("number of parameters for ansatz = ", num_params)


cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": []
}

def cost_func(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0] # lower eigenvalue
    # append to a global dict
    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    #
    return energy

x0 = 2*np.pi*np.random.random(num_params)


from math import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_aer import Aer
from qiskit import transpile
def run_on_statevec_sim(my_qc):
    backend = Aer.get_backend("statevector_simulator")
    tqc = transpile(my_qc, backend)
    job = backend.run(tqc)
    return job.result()


theta = Parameter("theta") # only one parameter
# more general use ParameterVector
qc = QuantumCircuit(2)
qc.x(0)
qc.barrier()
qc.rx(pi/2, 0) # negative sign (global phase)
qc.ry(pi/2, 1)
qc.cx(1, 0)
qc.rz(theta, 0)
qc.cx(1, 0)
qc.rx(pi/2, 0)
qc.ry(pi/2, 1)
qc.draw()


import numpy as np

def build_Hmol():
    g0 = -0.4804
    g1 = +0.3435
    g2 = -0.4347
    g3 = +0.5716
    g4 = +0.0910
    g5 = +0.0910
    # Pauli matrices
    I  = np.array([
        [ 1, 0],
        [ 0, 1]
    ])

    Sx = np.array([
        [ 0, 1],
        [ 1, 0]])
    Sy = np.array([
        [ 0,-1j],
        [1j, 0]
    ])
    Sz = np.array([
        [ 1, 0],
        [ 0,-1]
    ])
    # Build Hamiltonian matrix
    Hmol = (g0 * np.kron(I , I) + # g0 * I
            g1 * np.kron(I , Sz) + # g1 * Z0
            g2 * np.kron(Sz, I) + # g2 * Z1
            g3 * np.kron(Sz, Sz) + # g3 * Z0Z1
            g4 * np.kron(Sy, Sy) + # g4 * Y0Y1
            g5 * np.kron(Sx, Sx))  # g5 * X0X1
    return Hmol

E_nn = 0.7055696146 # at R=0.75A


# t_in must be an array
def calc_energy(t_in, qc, H_op):
    tqc = qc.assign_parameters(t_in)
    res = run_on_statevec_sim(tqc)
    psi = res.get_statevector().data # use np.array
    Hpsi = H_op @ psi
    energy = np.real(psi.conj().T @ Hpsi)
    return energy

from scipy.optimize import minimize
Hmol = build_Hmol()
theta0 = [0.0] # initial
min_result = minimize(calc_energy, theta0, args=(qc, Hmol), options={'disp': True})
theta  = min_result.x[0]
E_tot = min_result.fun + E_nn


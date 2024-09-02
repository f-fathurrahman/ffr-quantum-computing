import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize

#np.set_printoptions(precision=4,suppress=True)

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

# Hadamard matrix
H = (1/np.sqrt(2))*np.array([
    [ 1, 1],
    [ 1,-1]
])

# Phase matrix
S = np.array([
    [ 1, 0],
    [ 0,1j]
])

# single qubit basis states |0> and |1>
q0 = np.array([
    [1],
    [0]
])
q1 = np.array([
    [0],
    [1]
])

# Projection matrices |0><0| and |1><1|
P0  = np.dot(q0, q0.conj().T)
P1  = np.dot(q1, q1.conj().T)
# These are used in building CNOT gate


# Rotation matrices as a function of theta, e.g. Rx(theta), etc.
def Rx(θ):
    return np.array([
        [    np.cos(θ/2),-1j*np.sin(θ/2)],
        [-1j*np.sin(θ/2),    np.cos(θ/2)]
    ])

def Ry(θ):
    return np.array([
        [    np.cos(θ/2),   -np.sin(θ/2)],
        [    np.sin(θ/2),    np.cos(θ/2)]
    ])

def Rz(θ):
    return np.array([
        [np.exp(-1j*θ/2),            0.0],
        [            0.0, np.exp(1j*θ/2)]
    ])


# TODO: verify operations of these

# CNOTij, where i is control qubit and j is target qubit

# control -> q1, target -> q0
CNOT_10 = np.kron(P0, I) + np.kron(P1, Sx) 

# control -> q0, target -> q1  (Sx is in the q1 position)
CNOT_01 = np.kron(I, P0) + np.kron(Sx, P1)

SWAP   = block_diag(1, Sx, 1)


# See DOI: 10.1103/PhysRevX.6.031007
# Here, we use parameters given for H2 at R=0.75A
g0 = -0.4804
g1 = +0.3435
g2 = -0.4347
g3 = +0.5716
g4 = +0.0910
g5 = +0.0910

E_nn = 0.7055696146 # at R=0.75A

# Build Hamiltonian matrix
Hmol = (g0 * np.kron(I , I) + # g0 * I
        g1 * np.kron(I , Sz) + # g1 * Z0
        g2 * np.kron(Sz, I) + # g2 * Z1
        g3 * np.kron(Sz, Sz) + # g3 * Z0Z1
        g4 * np.kron(Sy, Sy) + # g4 * Y0Y1
        g5 * np.kron(Sx, Sx))  # g5 * X0X1

electronic_energy = np.linalg.eigvalsh(Hmol)[0] # take the lowest value
print("Classical diagonalization: {:+2.8} Eh".format(electronic_energy + E_nn))
print("Exact (from G16):          {:+2.8} Eh".format(-1.1457416808))

# initial basis, put in |01> state with Sx operator on q0
#psi0 = np.zeros((4,1))
#psi0[0] = 1
#psi0 = np.kron(I,Sx) @ psi0 # apply the X operator to q0

# Alternative
ket0 = np.array([[1], [0]])
ket1 = np.array([[0], [1]])
psi0 = np.kron(q0, q1) # State |01>


theta  = [0.0]
# The UCC ansatz in exponential form
from scipy.linalg import expm
def ansatz_UCC_expm(theta):
    return expm(-1j*np.array([theta])*np.kron(Sy,Sx))


# For a given ansatz with parameter theta, compute the expectation value
# We will minimize this function, the first arg will be varied during optimization
# via scipy.optimize.minimize
# The rest of args are not varied.
def expected(theta, ansatz, Hmol, psi0):
    circuit = ansatz(theta[0]) # the quantum circuit
    # apply the circuit operation to psi0
    # This will evolve psi0 -> psi
    psi = circuit @ psi0 
    # now compute the expectation with respect to Hmol operator
    Hpsi = Hmol @ psi
    psiHpsi = psi.conj().T @ Hpsi # psi is a (4x1) "matrix"
    return np.real(psiHpsi[0,0]) # because psiHpsi is (1x1) "matrix"
# XXX This only only works with expm ansatz


from scipy.optimize import minimize
theta  = [0.0]
result = minimize(expected, theta, args=(ansatz_UCC_expm, Hmol, psi0), options={'disp': True})
theta  = result.x[0]
E_tot = result.fun + E_nn

print()
print("VQE with UCC, using expm ")
print("theta:  {:+2.8} rad".format(theta))
print("energy: {:+2.8} Hartree".format(E_tot))


# read right-to-left (bottom-to-top?)
# ansatz in terms of gate operations
def ansatz_UCC_v1(theta):
    # remember that we use |q1 q0>
    gate5 = np.kron(-Ry(np.pi/2), Rx(np.pi/2))
    gate4 = CNOT_10   # control -> q1, target -> q0
    gate3 = np.kron(I, Rz(theta))
    gate2 = CNOT_10
    gate1 = np.kron(Ry(np.pi/2), -Rx(np.pi/2))
    return gate1 @ gate2 @ gate3 @ gate4 @ gate5

def ansatz_UCC_v2(theta):
    # remember that we use |q1 q0>
    gate5 = np.kron(-Ry(np.pi/2), Rx(np.pi/2))
    gate4 = CNOT_10   # control -> q1, target -> q0
    gate3 = np.kron(I, Rz(theta))
    gate2 = CNOT_10
    gate1 = np.kron(Ry(np.pi/2), -Rx(np.pi/2))
    return gate5 @ gate4 @ gate3 @ gate2 @ gate1 # psi



def ansatz_UCC_orig(theta):
    return ( np.dot(np.dot(np.kron(-Ry(np.pi/2), Rx(np.pi/2)),
             np.dot(CNOT_10, 
                np.dot(np.kron(I,Rz(theta)),
                CNOT_10))),
                np.kron(Ry(np.pi/2),-Rx(np.pi/2)))
           )
    

# XXX this function need access to some global variables
# XXX make Pauli strings as input argument
def projective_expected(theta, ansatz, psi0):
    # this will depend on the hard-coded Hamiltonian + coefficients
    circuit = ansatz(theta[0])
    psi = np.dot(circuit, psi0)
    
    # for 2 qubits, assume we can only take Pauli Sz measurements (Sz ⊗ I)
    # we just apply the right unitary for the desired Pauli measurement
    measureZ = lambda U: np.dot(np.conj(U).T, np.dot(np.kron(Sz,I),U))
    
    energy = 0.0
    
    # although the paper indexes the hamiltonian left-to-right (0-to-1) 
    # qubit-1 is always the top qubit for us, so the tensor pdt changes
    # e.g. compare with the "exact Hamiltonian" we explicitly diagonalized
    
    # <I1 I0> 
    energy += g0 # it is a constant

    # <I1 Sz0>
    U = SWAP
    energy += g1*np.dot(psi.conj().T,np.dot(measureZ(U),psi))

    # <Sz1 I0>
    U = np.kron(I,I)
    energy += g2*np.dot(psi.conj().T,np.dot(measureZ(U),psi))

    # <Sz1 Sz0>
    U = CNOT_01
    energy += g3*np.dot(psi.conj().T,np.dot(measureZ(U),psi))

    # <Sx1 Sx0>
    U = np.dot(CNOT_01,np.kron(H,H))
    energy += g4*np.dot(psi.conj().T,np.dot(measureZ(U),psi))

    # <Sy1 Sy0>
    U = np.dot(CNOT_01,np.kron(np.dot(H,S.conj().T),np.dot(H,S.conj().T)))
    energy += g5*np.dot(psi.conj().T,np.dot(measureZ(U),psi))

    return np.real(energy)[0,0]


theta  = [0.0] # we are using `minimize` for vector
result = minimize(projective_expected, theta, args=(ansatz_UCC_orig, psi0))
theta  = result.x[0]
E_tot = result.fun + E_nn

# check it works...
assert np.allclose(E_tot, -1.1456295)

print()
print("VQE with quantum circuit: ")
print(" theta:  {:+2.8} deg".format(theta))
print(" energy: {:+2.8} Hartree".format(E_tot))

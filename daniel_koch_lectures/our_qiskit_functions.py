from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer

import math
import numpy as np

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

USING_QISKIT_CONVENTION = False

def toBinary(number, total):
    Nqubits = int( math.log2(total) )
    N = number
    b_num = np.zeros(Nqubits)
    for i in range(Nqubits):
        if N/( 2**(Nqubits-i-1)) >= 1:
            b_num[i] = 1
            N = N - 2**(Nqubits-i-1)
    B = []
    for j in range(Nqubits):
        B.append(int(b_num[j]))
    if not USING_QISKIT_CONVENTION:
        # reverse the list if not using Qiskit convention
        B.reverse()
    return B


def Wavefunction(obj, *args, **kwargs):
    #
    if type(obj) == QuantumCircuit:
        job = execute(obj, S_simulator, shots=1)
        statevec = np.asarray(job.result().get_statevector())
        # It is best to cast the result to np.array to get around deprecation
    #
    if type(obj) == np.ndarray:
        statevec = obj
    #
    #print("statevec = ")
    #print(statevec)
    #print()
    #
    is_sys = False
    new_line = False
    dec = 5
    #
    wavefunction = ""
    Nqubits = int(math.log2(len(statevec)))
    #
    for i in range(len(statevec)):
        # Round the coefficient
        value = round(statevec[i].real, dec) + round(statevec[i].imag, dec)*1j
        if (value.real != 0) or (value.imag != 0):
            state = list( toBinary(i, 2**Nqubits) )
            #print("state = ", state)
            state_str = ""
            # is_sys == true is skipped
            for j in range(len(state)):
                if type(state[j]) != str:
                    state_str += str(int(state[j])) # FIXME need int?
                else:
                    state_str += state[j]
            #
            #print("outside_loop: state_str = ", state_str)
            # Both real and imag components are not zero
            if (value.real != 0) and (value.imag != 0):
                if value.imag > 0:
                    wavefunction += str(value.real) + ' + ' + str(value.imag) + 'j |' + state_str + '>    '
                else:
                    wavefunction += str(value.real) + '' + str(value.imag) + 'j |' + state_str + '>    '
            #
            # Pure real case
            #
            if (value.real != 0) and (value.imag == 0):
                wavefunction += str(value.real) + ' |' + state_str + '>    '
            #
            # Pure imaginary case
            # 
            if (value.real == 0) and (value.imag != 0):
                wavefunction += str(value.imag) + 'j |' + state_str + '>    '
            if new_line:
                wavefunction += '\n'

    print(wavefunction)

def measure_pauli_1q(circuit,index,pauli=None):
    if pauli == 'I' or pauli == 'Z':
        circuit.id([index])
    elif pauli == 'X':
        circuit.h([index])
    elif pauli == 'Y':
        circuit.s([index])
        circuit.s([index])
        circuit.s([index])
        circuit.h([index])
    else:
        assert 1==0

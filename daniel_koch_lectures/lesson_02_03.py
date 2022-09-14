from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute

import numpy as np

S_simulator = Aer.backends(name="statevector_simulator")[0]
M_simulator = Aer.backends(name="qasm_simulator")[0]

q = QuantumRegister(2, name="q")
c = ClassicalRegister(2, name="c")
qc = QuantumCircuit(q, c, name="qc")

qc.h(q[0])
qc.h(q[1])

qc.measure(q[0], c[0])

str_instr = qc.qasm()
print(str_instr[36:len(str_instr)])

instr1 = qc.data[1] # make a copy

#del qc.data[1]
#print("After deleting qc.data[1]")
#str_instr = qc.qasm()
#print(str_instr[36:len(str_instr)])

#qc.data.append(instr1)
#print("After append instr1")
#str_instr = qc.qasm()
#print(str_instr[36:len(str_instr)])


#qc.data.insert(0, instr1)
#print("After inserting instr1 at location 0")
#str_instr = qc.qasm()
#print(str_instr[36:len(str_instr)])


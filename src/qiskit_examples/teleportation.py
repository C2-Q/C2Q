"""Example of quantum teleportation for a single qubit."""

from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(3, 3)
qc.h(1)
qc.cx(1, 2)

qc.cx(0, 1)
qc.h(0)
qc.measure([0, 1], [0, 1])

qc.cx(1, 2).c_if(qc.clbits[1], 1)
qc.z(2).c_if(qc.clbits[0], 1)

qc.measure(2, 2)
backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend).result()
print('Teleportation counts:', result.get_counts())

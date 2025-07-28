"""Ripple-carry adder for two 2-bit numbers."""
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(6, 3)

# Inputs a1 a0 and b1 b0
qc.x(1)  # a0=1
qc.x(3)  # b0=1
qc.cx(1, 2)
qc.cx(3, 2)
qc.ccx(1, 3, 2)
qc.cx(0, 4)
qc.cx(2, 4)
qc.ccx(0, 2, 4)
qc.cx(1, 5)
qc.cx(3, 5)
qc.ccx(1, 3, 5)

qc.measure([4, 5, 2], [0, 1, 2])
backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend, shots=1).result()
print(result.get_counts())

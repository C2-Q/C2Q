"""Simple 2-bit multiplication using controlled additions."""
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(8, 4)

# Inputs a1 a0 and b1 b0
qc.x(1)  # a0=1
qc.x(3)  # b0=1

qc.ccx(1, 3, 4)
qc.ccx(1, 2, 5)
qc.ccx(0, 3, 6)
qc.ccx(0, 2, 7)

qc.measure([4, 5, 6, 7], [0, 1, 2, 3])
backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend, shots=1).result()
print(result.get_counts())

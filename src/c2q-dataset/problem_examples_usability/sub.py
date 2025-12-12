"""Subtract two 2-bit numbers using two's complement."""
from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(6, 3)

# a1 a0 minus b1 b0
qc.x([1, 3])  # set a0=1 and b0=1

# Two's complement of b
qc.x(2)
qc.x(3)
qc.cx(3, 2)
qc.cx(2, 3)
qc.cx(3, 2)

# Add a and complemented b
qc.cx(1, 4)
qc.cx(3, 4)
qc.ccx(1, 3, 4)
qc.cx(0, 5)
qc.cx(2, 5)
qc.ccx(0, 2, 5)

qc.measure([4, 5, 2], [0, 1, 2])
backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend, shots=1).result()
print(result.get_counts())

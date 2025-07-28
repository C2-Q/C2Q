"""Simple three-qubit bit-flip code."""

from qiskit import QuantumCircuit, Aer, execute

qc = QuantumCircuit(3, 1)

# Encode |1>
qc.x(0)
qc.cx(0, 1)
qc.cx(0, 2)

# Introduce an error on qubit 1
qc.x(1)

# Decode
qc.cx(0, 1)
qc.cx(0, 2)
qc.ccx(1, 2, 0)

qc.measure(0, 0)
backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend).result()
print('Error correction counts:', result.get_counts())

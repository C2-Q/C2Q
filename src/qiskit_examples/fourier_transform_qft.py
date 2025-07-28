"""Demonstrate the Quantum Fourier Transform on 3 qubits."""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import QFT

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)

qft = QFT(3)
qc.append(qft, [0,1,2])
qc.measure_all()

backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend).result()
print('QFT circuit counts:', result.get_counts())

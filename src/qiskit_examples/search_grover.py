"""Grover's algorithm for a simple oracle returning '11'."""

from qiskit import QuantumCircuit, Aer, assemble
from qiskit.circuit.library import GroverOperator

oracle = QuantumCircuit(2)
oracle.cz(0, 1)
oracle = GroverOperator(oracle)

qc = QuantumCircuit(2)
qc.h([0, 1])
qc.append(oracle, [0, 1])
qc.measure_all()

backend = Aer.get_backend('aer_simulator')
qobj = assemble(qc)
result = backend.run(qobj).result()
counts = result.get_counts()
print('Measurement results:', counts)

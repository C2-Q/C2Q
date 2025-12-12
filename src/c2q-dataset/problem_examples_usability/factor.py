"""Shor's algorithm for factoring 15 implemented without qiskit.algorithms."""
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import QFT

# Order finding for a=2, N=15
n_count = 4
qc = QuantumCircuit(n_count + 4, n_count)

for q in range(n_count):
    qc.h(q)

qc.x(n_count)

for j in range(n_count):
    repetitions = 2**j
    for _ in range(repetitions):
        qc.cx(n_count, n_count+1)
        qc.cx(n_count+1, n_count+2)
        qc.cx(n_count+2, n_count+3)

qft = QFT(n_count).inverse()
qc.append(qft.to_instruction(), range(n_count))
qc.measure(range(n_count), range(n_count))

backend = Aer.get_backend('aer_simulator')
result = execute(qc, backend, shots=1024).result()
print(result.get_counts())

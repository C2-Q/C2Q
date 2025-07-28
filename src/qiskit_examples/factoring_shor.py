"""Run Shor's algorithm to factor 15."""

from qiskit.algorithms import Shor
from qiskit import Aer

backend = Aer.get_backend('aer_simulator')
shor = Shor(quantum_instance=backend)
result = shor.factor(15)
print('Factors of 15:', result.factors)

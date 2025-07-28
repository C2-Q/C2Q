"""Solve a simple system of linear equations using HHL."""

from qiskit import Aer
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.algorithms.linear_solvers.matrices import Matrix
from qiskit.circuit.library import QFT

matrix = Matrix([[1, -1/3], [-1/3, 1]])
vector = [1, 0]

hhl = HHL(qubit_converter=None, quantum_instance=Aer.get_backend('aer_simulator_statevector'))
solution = hhl.solve(matrix, vector)
print('Solution:', solution.state)

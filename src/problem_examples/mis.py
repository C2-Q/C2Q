"""Solve a simple Maximum Independent Set using QAOA from a QUBO."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# 2-node graph with one edge
qp = QuadraticProgram()
qp.binary_var("x0")
qp.binary_var("x1")
# Objective: maximise x0 + x1 with penalty for selecting adjacent nodes
A = 2
qp.minimize(linear=[-1, -1], quadratic={(0, 1): A})

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Independent set assignment:", result.x)
print("Objective value:", result.fval)

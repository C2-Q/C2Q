"""Example QAOA implementation for Max-Cut problem using Qiskit."""

from qiskit import Aer, execute
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Simple 2-node Max-Cut graph
qp = QuadraticProgram()
qp.binary_var('x0')
qp.binary_var('x1')
qp.minimize(linear=[0, 0], quadratic={(0, 1): -1})

# Setup QAOA
backend = Aer.get_backend('qasm_simulator')
ansatz = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
optimizer = MinimumEigenOptimizer(qaoa)

# Solve
result = optimizer.solve(qp)
print('Optimal solution:', result.x)
print('Optimal value:', result.fval)

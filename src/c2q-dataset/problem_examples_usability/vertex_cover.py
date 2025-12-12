"""Minimum vertex cover on a triangle expressed as a QUBO."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# binary variable per node indicating if it is in the cover
qp = QuadraticProgram()
for i in range(3):
    qp.binary_var(f"x{i}")

B = 2  # penalty for uncovered edge
linear = [1 - 2*B, 1 - 2*B, 1 - 2*B]
quadratic = {(0, 1): B, (0, 2): B, (1, 2): B}
qp.minimize(linear=linear, quadratic=quadratic)

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Vertex cover assignment:", result.x)
print("Objective:", result.fval)

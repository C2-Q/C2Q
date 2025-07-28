"""2-coloring of a triangle graph using a QUBO and QAOA."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# one binary var per node representing color 0 or 1
qp = QuadraticProgram()
for i in range(3):
    qp.binary_var(f"x{i}")

# penalty if adjacent nodes share the same color
quadratic = {(0, 1): 2, (0, 2): 2, (1, 2): 2}
linear = [-2, -2, -2]
qp.minimize(linear=linear, quadratic=quadratic)

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Color assignment:", result.x)
print("Objective:", result.fval)

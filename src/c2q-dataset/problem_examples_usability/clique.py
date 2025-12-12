"""Find a 2-node clique in a 3-node graph using a QUBO formulation."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Complete graph on 3 nodes looking for clique of size 2
A = 2
B = 1
qp = QuadraticProgram()
for i in range(3):
    qp.binary_var(f"x{i}")
linear = [A * (1 - 4)] * 3  # K=2
quadratic = {(0, 1): 2 * A - B, (0, 2): 2 * A - B, (1, 2): 2 * A - B}
qp.minimize(linear=linear, quadratic=quadratic)

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Clique assignment:", result.x)
print("Objective value:", result.fval)

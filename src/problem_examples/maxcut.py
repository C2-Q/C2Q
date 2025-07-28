"""QAOA solution of MaxCut on a triangle using an explicit QUBO matrix."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Triangle graph QUBO: minimise -x0*x1 - x0*x2 - x1*x2
qp = QuadraticProgram()
for i in range(3):
    qp.binary_var(name=f"x{i}")
qp.minimize(linear=[0, 0, 0], quadratic={(0, 1): -1, (0, 2): -1, (1, 2): -1})

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Optimal assignment:", result.x)
print("Optimal cut value:", -result.fval)

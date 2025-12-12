"""Tiny 3-city TSP formulated as a QUBO and solved with QAOA."""

from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.circuit.library import TwoLocal
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

n = 3
# symmetric distance matrix
dist = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
A = 3
Q = np.zeros((n * n, n * n))

# Distance costs
for i in range(n):
    for j in range(n):
        if i != j:
            w = dist[i, j]
            for p in range(n - 1):
                Q[i*n + p, j*n + p + 1] += w

# Penalties so each city appears once
for v in range(n):
    for j in range(n):
        Q[v*n + j, v*n + j] += -A
        for k in range(j + 1, n):
            Q[v*n + j, v*n + k] += 2 * A

# Penalties so each position is occupied
for j in range(n):
    for v in range(n):
        Q[v*n + j, v*n + j] += -A
        for u in range(v + 1, n):
            Q[v*n + j, u*n + j] += 2 * A

qp = QuadraticProgram()
for i in range(n*n):
    qp.binary_var(name=f"x{i}")
quad = {(i, j): Q[i, j] for i in range(n*n) for j in range(i, n*n) if Q[i, j] != 0}
qp.minimize(linear=[0]*(n*n), quadratic=quad)

backend = Aer.get_backend("qasm_simulator")
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
qaoa = QAOA(optimizer=None, reps=1, quantum_instance=backend)
solver = MinimumEigenOptimizer(qaoa)
result = solver.solve(qp)

print("Binary solution:", result.x)
print("Objective:", result.fval)

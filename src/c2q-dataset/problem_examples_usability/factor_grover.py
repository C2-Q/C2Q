from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -----------------------------
# TSP setup (3 cities)
# -----------------------------

dist_matrix = np.array([
    [0, 1, 3],
    [1, 0, 2],
    [3, 2, 0]
])

n = 3  # Number of cities
num_qubits = n * n  # each city visited at each time step

backend = AerSimulator()

# -----------------------------
# Helper functions
# -----------------------------

def index(city, pos):
    return city * n + pos

def create_cost_operator():
    """Construct diagonal cost operator as np.array."""
    cost = np.zeros(2**num_qubits)
    for i in range(2**num_qubits):
        bitstr = format(i, f"0{num_qubits}b")
        matrix = np.array([[int(bitstr[index(c, t)]) for t in range(n)] for c in range(n)])
        # Each city visited once & once per time step
        if np.all(matrix.sum(axis=0) == 1) and np.all(matrix.sum(axis=1) == 1):
            path = np.argmax(matrix, axis=0)
            d = sum(dist_matrix[path[i], path[(i + 1) % n]] for i in range(n))
            cost[i] = d
        else:
            cost[i] = 999  # Penalize invalid routes
    return cost

cost_operator = create_cost_operator()

# -----------------------------
# QAOA Circuit
# -----------------------------

def build_qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(num_qubits)

    # Initial state: Hadamards
    qc.h(range(num_qubits))

    # Cost unitary
    for i, coeff in enumerate(cost_operator):
        if coeff != 0:
            bin_str = format(i, f"0{num_qubits}b")
            z_indices = [j for j, bit in enumerate(bin_str) if bit == '1']
            if z_indices:
                qc.rz(2 * gamma * coeff, z_indices[-1])  # simplified Z rotations

    # Mixer unitary
    for q in range(num_qubits):
        qc.rx(2 * beta, q)

    return qc

# -----------------------------
# Expectation Function
# -----------------------------

def expectation(counts):
    exp = 0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        i = int(bitstring, 2)
        exp += cost_operator[i] * count / total
    return exp

def evaluate(params):
    gamma, beta = params
    qc = build_qaoa_circuit(gamma, beta)
    qc.measure_all()
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=1024).result()
    counts = result.get_counts()
    return expectation(counts)

# -----------------------------
# Optimize Parameters
# -----------------------------

initial_params = [0.5, 0.5]
res = minimize(evaluate, initial_params, method="COBYLA")
optimal_gamma, optimal_beta = res.x

# -----------------------------
# Final Execution
# -----------------------------

final_qc = build_qaoa_circuit(optimal_gamma, optimal_beta)
final_qc.measure_all()
transpiled = transpile(final_qc, backend)
counts = backend.run(transpiled, shots=1024).result().get_counts()

print("Most probable bitstring:", max(counts, key=counts.get))
plot_histogram(counts)
plt.show()
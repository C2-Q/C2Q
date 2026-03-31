from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Problem Setup
# -----------------------------

num_qubits = 5
adj_matrix = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
])
backend = AerSimulator()
shots = 1024


# -----------------------------
# Utility Functions
# -----------------------------

def is_valid_mis(bitstring, adj):
    """Check if a bitstring represents a valid independent set."""
    for i in range(len(bitstring)):
        if bitstring[i] == '1':
            for j in range(i + 1, len(bitstring)):
                if bitstring[j] == '1' and adj[i][j] == 1:
                    return False
    return True


def count_ones(bitstring):
    return bitstring.count('1')


# -----------------------------
# Oracle and Diffuser
# -----------------------------

def build_oracle(adj):
    qc = QuantumCircuit(num_qubits)

    # Reward nodes individually
    for i in range(num_qubits):
        qc.z(i)

    # Penalize edges (invalid combinations)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if adj[i][j] == 1:
                qc.cz(i, j)

    return qc


def build_diffuser():
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_initial_state():
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    return qc


# -----------------------------
# Grover Iteration
# -----------------------------

def run_grover(adj):
    oracle = build_oracle(adj)
    diffuser = build_diffuser()

    m = 1
    λ = 8 / 7
    best_solution = ""
    max_size = 0

    while m <= 2 ** (num_qubits // 2):
        j = np.random.randint(0, m)
        qc = QuantumCircuit(num_qubits, num_qubits)

        qc.compose(build_initial_state(), inplace=True)

        for _ in range(j):
            qc.compose(oracle, inplace=True)
            qc.compose(diffuser, inplace=True)

        qc.measure(range(num_qubits), range(num_qubits))
        transpiled = transpile(qc, backend)
        result = backend.run(transpiled, shots=shots).result()
        counts = result.get_counts()

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for bitstring, _ in sorted_counts:
            bitstring = bitstring[::-1]  # reverse due to endianness
            if is_valid_mis(bitstring, adj):
                size = count_ones(bitstring)
                if size > max_size:
                    max_size = size
                    best_solution = bitstring
                break

        if best_solution:
            break

        m = min(int(np.ceil(λ * m)), int(np.ceil(np.sqrt(2 ** num_qubits))))

    return best_solution, counts


# -----------------------------
# Execution
# -----------------------------

solution, counts = run_grover(adj_matrix)
print(f"Best valid independent set: {solution} (size = {count_ones(solution)})")
plot_histogram(counts)
plt.show()
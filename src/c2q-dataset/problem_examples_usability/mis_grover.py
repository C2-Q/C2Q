from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.visualization import plot_histogram
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Problem setup and parameters
# -----------------------------

num_qubits = 5
shots = 1024
MAX_ITER = 3
backend = Aer.get_backend('qasm_simulator')

# Adjacency matrix of the graph (5 nodes)
adj_matrix = np.array([
    [0,1,1,0,0],
    [1,0,1,1,0],
    [1,1,0,0,1],
    [0,1,0,0,1],
    [0,0,1,1,0]
])

# -----------------------------
# Helper Functions
# -----------------------------

def is_valid_mis(bitstring, adj):
    """Check if bitstring is a valid independent set."""
    for i in range(len(bitstring)):
        if bitstring[i] == '1':
            for j in range(i+1, len(bitstring)):
                if bitstring[j] == '1' and adj[i][j] == 1:
                    return False
    return True

def count_ones(bitstring):
    return bitstring.count('1')

# -----------------------------
# Circuit Construction
# -----------------------------

def build_initial_state():
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    return qc

def build_mis_oracle(adj):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.z(i)  # reward individual node
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            if adj[i][j]:
                qc.cz(i, j)  # penalize edge conflicts
    return qc

def build_diffusion_operator():
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits-1)
    qc.mcx(list(range(num_qubits-1)), num_qubits-1)
    qc.h(num_qubits-1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc

# -----------------------------
# Grover Search
# -----------------------------

def run_grover(adj):
    oracle = build_mis_oracle(adj)
    diffuser = build_diffusion_operator()

    m = 1
    lam = 8/7
    found = False
    best_bitstring = ""

    while m <= 2**(num_qubits//2):
        j = np.random.randint(0, m)
        qc = QuantumCircuit(num_qubits, num_qubits)

        qc += build_initial_state()

        for _ in range(j):
            qc += oracle
            qc += diffuser

        qc.measure(range(num_qubits), range(num_qubits))
        transpiled = transpile(qc, backend=backend, optimization_level=3)
        result = execute(transpiled, backend=backend, shots=shots).result()
        counts = result.get_counts()

        # Sort and filter valid MIS
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for bitstring, count in sorted_counts:
            if is_valid_mis(bitstring[::-1], adj):  # Reverse due to endianness
                if count_ones(bitstring) > count_ones(best_bitstring):
                    best_bitstring = bitstring
                    found = True
                    break

        if found:
            break
        m = min(int(np.ceil(lam * m)), int(np.ceil(np.sqrt(2**num_qubits))))

    return best_bitstring[::-1], counts

# -----------------------------
# Execute and Interpret
# -----------------------------

solution, all_counts = run_grover(adj_matrix)

print(f"Best valid independent set (bitstring): {solution}")
print(f"Set size: {count_ones(solution)}")

# Plot full histogram
plot_histogram(all_counts, title="Measurement Outcomes")
plt.show()
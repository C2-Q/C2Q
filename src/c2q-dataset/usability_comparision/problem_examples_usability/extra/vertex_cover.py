from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
from scipy.optimize import minimize
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# -----------------------------
# Problem: Minimum Vertex Cover
# -----------------------------
# Graph: 3 nodes, edges = (0,1), (1,2)
edges = [(0, 1), (1, 2)]
num_nodes = 3
num_qubits = num_nodes
backend = AerSimulator()

# -----------------------------
# Cost Hamiltonian
# -----------------------------
# Cost = # of selected vertices (sum of Zs) + large penalty if any edge not covered

def cost_bitstring(bitstring):
    """Evaluate cost of a bitstring for Vertex Cover."""
    bitstring = bitstring[::-1]  # Qiskit ordering
    selected = [i for i, b in enumerate(bitstring) if b == '1']
    penalty = 0
    for u, v in edges:
        if u not in selected and v not in selected:
            penalty += 10  # large penalty
    return len(selected) + penalty

# Precompute diagonal Hamiltonian (classical simulation of QAOA)
cost_diag = np.zeros(2 ** num_qubits)
for i in range(2 ** num_qubits):
    bitstr = format(i, f"0{num_qubits}b")
    cost_diag[i] = cost_bitstring(bitstr)

# -----------------------------
# QAOA Circuit
# -----------------------------

def build_qaoa_circuit(gamma, beta):
    qc = QuantumCircuit(num_qubits)

    # Initial state
    qc.h(range(num_qubits))

    # Cost unitary (apply RZ rotations based on Z)
    for i, cost in enumerate(cost_diag):
        if cost != 0:
            bits = format(i, f"0{num_qubits}b")
            z_indices = [j for j, b in enumerate(bits) if b == '1']
            if z_indices:
                qc.rz(2 * gamma * cost, z_indices[-1])  # simplified

    # Mixer unitary
    for q in range(num_qubits):
        qc.rx(2 * beta, q)

    return qc

# -----------------------------
# Expectation Function
# -----------------------------

def evaluate_counts(counts):
    total = sum(counts.values())
    expected = 0
    for bitstring, count in counts.items():
        i = int(bitstring, 2)
        expected += cost_diag[i] * count / total
    return expected

def energy(params):
    gamma, beta = params
    qc = build_qaoa_circuit(gamma, beta)
    qc.measure_all()
    transpiled = transpile(qc, backend)
    result = backend.run(transpiled, shots=1024).result()
    counts = result.get_counts()
    return evaluate_counts(counts)

# -----------------------------
# Optimize QAOA Parameters
# -----------------------------

init = [0.5, 0.5]
res = minimize(energy, init, method='COBYLA')
opt_gamma, opt_beta = res.x

# -----------------------------
# Final Circuit Execution
# -----------------------------

final_qc = build_qaoa_circuit(opt_gamma, opt_beta)
final_qc.measure_all()
final_transpiled = transpile(final_qc, backend)
result = backend.run(final_transpiled, shots=1024).result()
counts = result.get_counts()

print("Most probable bitstring:", max(counts, key=counts.get))
plot_histogram(counts)
plt.show()
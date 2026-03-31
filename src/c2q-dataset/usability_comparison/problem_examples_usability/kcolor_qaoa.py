from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Problem Setup
# -----------------------------
n_nodes = 4
k_colors = 3
backend = AerSimulator()

# Sample graph (undirected, adjacency matrix)
adj_matrix = np.array([
    [0,1,1,0],
    [1,0,1,1],
    [1,1,0,1],
    [0,1,1,0]
])

# -----------------------------
# QUBO Construction
# -----------------------------

def get_kcolor_qubo(adj, k):
    n = len(adj)
    N = n * k  # total variables
    Q = np.zeros((N, N))

    # One color per node constraint: sum_{c} x_{v,c} = 1
    for v in range(n):
        for c in range(k):
            i = v * k + c
            Q[i][i] += -2
            for c2 in range(c+1, k):
                j = v * k + c2
                Q[i][j] += 2
                Q[j][i] += 2

    # Adjacent nodes can't share same color: x_{u,c} * x_{v,c}
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v] == 1:
                for c in range(k):
                    i = u * k + c
                    j = v * k + c
                    Q[i][j] += 1
                    Q[j][i] += 1
    return Q

qubo = get_kcolor_qubo(adj_matrix, k_colors)

# -----------------------------
# QUBO to Ising
# -----------------------------

def qubo_to_ising(Q):
    n = Q.shape[0]
    pauli_terms = []
    offset = 0
    for i in range(n):
        offset += Q[i][i] / 4
        if Q[i][i] != 0:
            z = ['I'] * n
            z[i] = 'Z'
            pauli_terms.append(("".join(z), Q[i][i] / 2))
        for j in range(i+1, n):
            if Q[i][j] != 0:
                z = ['I'] * n
                z[i] = 'Z'
                z[j] = 'Z'
                pauli_terms.append(("".join(z), Q[i][j] / 4))
                offset += Q[i][j] / 4
    return SparsePauliOp.from_list(pauli_terms), offset

ising_hamiltonian, offset = qubo_to_ising(qubo)
n_qubits = k_colors * n_nodes

# -----------------------------
# QAOA Setup
# -----------------------------
p = 1
params = ParameterVector("θ", 2*p)

def build_qaoa_circuit(n, hamiltonian, params):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()

    for i in range(p):
        gamma = params[2*i]
        beta = params[2*i + 1]
        qc.append(PauliEvolutionGate(hamiltonian, time=gamma), range(n))
        qc.barrier()
        qc.rx(2 * beta, range(n))
        qc.barrier()

    qc.measure_all()
    return qc

# -----------------------------
# Optimization & Execution
# -----------------------------
estimator = Estimator(mode=backend)
estimator.options.default_shots = 1024
init_params = [2*np.pi, np.pi] * p
exp_vals = []

def cost_fn(theta):
    qc = build_qaoa_circuit(n_qubits, ising_hamiltonian, params)
    qc_bound = qc.assign_parameters(dict(zip(params, theta)))
    transpiled = transpile(qc_bound, backend)
    job = estimator.run([(transpiled, ising_hamiltonian)])
    val = job.result()[0].data.evs
    exp_vals.append(val)
    return val

opt_result = minimize(cost_fn, init_params, method="Powell", options={"maxiter": 300})
opt_theta = opt_result.x

# -----------------------------
# Sampling from Final Circuit
# -----------------------------
final_qc = build_qaoa_circuit(n_qubits, ising_hamiltonian, params)
final_qc = final_qc.assign_parameters(dict(zip(params, opt_theta)))
final_qc = transpile(final_qc, backend)

sampler = Sampler(mode=backend)
sampler.options.default_shots = 1024
result = sampler.run([final_qc]).result()
counts = result[0].data.meas.get_counts()

# -----------------------------
# Result Analysis
# -----------------------------
def interpret_solution(bitstring, n, k):
    color_map = {}
    valid = True
    for v in range(n):
        colors = bitstring[v*k:(v+1)*k]
        if colors.count('1') != 1:
            valid = False
        color_map[v] = colors.index('1') if '1' in colors else -1
    return color_map, valid

best = max(counts.items(), key=lambda x: x[1])[0]
coloring, is_valid = interpret_solution(best[::-1], n_nodes, k_colors)

print("Most frequent bitstring:", best)
print("Valid coloring?", is_valid)
print("Color assignment:", coloring)

# Plot
plt.bar(counts.keys(), counts.values())
plt.title("QAOA for K-Coloring")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
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
# Define a 4-node graph
adj_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])
n = len(adj_matrix)

# QUBO matrix for maximum clique (maximize sum x_i - sum x_ix_j for non-edges)
qubo = np.zeros((n, n))
for i in range(n):
    qubo[i][i] = 1  # reward inclusion
for i in range(n):
    for j in range(i + 1, n):
        if adj_matrix[i][j] == 0:
            qubo[i][j] = -1
            qubo[j][i] = -1

# -----------------------------
# QUBO to Ising
# -----------------------------
def convert_qubo_to_ising(qubo):
    n = len(qubo)
    pauli_list = []
    offset = 0
    for i in range(n):
        for j in range(i, n):
            z = ["I"] * n
            if i == j:
                z[i] = "Z"
                coeff = -(1/2) * qubo[i][i] - (1/4) * np.sum(qubo[i][i+1:]) - (1/4) * np.sum(qubo[:i, i])
                offset += 0.5 * qubo[i][i]
            else:
                z[i] = "Z"
                z[j] = "Z"
                coeff = 0.25 * qubo[i][j]
                offset += 0.25 * qubo[i][j]
            if coeff != 0:
                pauli_list.append(("".join(z), coeff))
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    return hamiltonian, offset

ising, offset = convert_qubo_to_ising(qubo)

# -----------------------------
# QAOA Construction
# -----------------------------
p = 1
parameters = ParameterVector('theta', 2 * p)

def build_qaoa_circuit(n, hamiltonian, parameters):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()

    for i in range(p):
        gamma = parameters[2*i]
        beta = parameters[2*i + 1]

        qc.append(PauliEvolutionGate(hamiltonian, time=gamma), range(n))
        qc.barrier()
        qc.rx(2 * beta, range(n))
        qc.barrier()

    qc.measure_all()
    return qc

# -----------------------------
# Estimation and Optimization
# -----------------------------
backend = AerSimulator()
estimator = Estimator(mode=backend)
estimator.options.default_shots = 1024

initial_theta = [2 * np.pi, np.pi] * p
exp_values = []

def cost_function(theta):
    qc = build_qaoa_circuit(n, ising, parameters)
    qc_bound = qc.assign_parameters(dict(zip(parameters, theta)))
    transpiled = transpile(qc_bound, backend)
    job = estimator.run([(transpiled, ising)])
    val = job.result()[0].data.evs
    exp_values.append(val)
    return val

opt_result = minimize(cost_function, initial_theta, method="Powell", options={"maxiter": 200})
opt_theta = opt_result.x

# -----------------------------
# Final Sampling
# -----------------------------
qc_final = build_qaoa_circuit(n, ising, parameters)
qc_final = qc_final.assign_parameters(dict(zip(parameters, opt_theta)))
qc_final = transpile(qc_final, backend)

sampler = Sampler(mode=backend)
sampler.options.default_shots = 1024
result = sampler.run([qc_final]).result()
counts = result[0].data.meas.get_counts()

# -----------------------------
# Print and Plot Results
# -----------------------------
def clique_size(bitstring, adj):
    bits = [int(b) for b in bitstring]
    indices = [i for i, b in enumerate(bits) if b == 1]
    for i in indices:
        for j in indices:
            if i != j and adj[i][j] == 0:
                return -1  # not a clique
    return len(indices)

best = max(counts.items(), key=lambda x: x[1])[0]
size = clique_size(best[::-1], adj_matrix)  # reverse due to Qiskit endianness

print("Most frequent bitstring:", best)
print("Clique size:", size)
print("Counts:", counts)

plt.bar(counts.keys(), counts.values())
plt.title("QAOA for Clique")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
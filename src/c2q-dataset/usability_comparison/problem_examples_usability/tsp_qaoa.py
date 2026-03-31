from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import Estimator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
import numpy as np

# -----------------------------
# Problem Setup
# -----------------------------

n_cities = 3
num_qubits = n_cities * n_cities  # x[i][j]: city i at position j
distance_matrix = np.array([
    [0, 1, 2],
    [1, 0, 3],
    [2, 3, 0]
])

# -----------------------------
# Helper: Build QUBO for TSP
# -----------------------------

def tsp_qubo(dist, n):
    Q = np.zeros((n*n, n*n))

    # 1. Each city appears once in the tour
    A = 10
    for i in range(n):
        for j in range(n):
            Q[i*n + j][i*n + j] -= 2 * A
            for k in range(j + 1, n):
                Q[i*n + j][i*n + k] += 2 * A

    # 2. Each position is occupied by one city
    for j in range(n):
        for i in range(n):
            Q[i*n + j][i*n + j] -= 2 * A
            for k in range(i + 1, n):
                Q[i*n + j][k*n + j] += 2 * A

    # 3. Minimize distance cost
    B = 1
    for i in range(n):
        for j in range(n):
            if i != j:
                for t in range(n):
                    from_idx = i*n + t
                    to_idx = j*n + ((t+1) % n)
                    Q[from_idx][to_idx] += B * dist[i][j]
    return Q

qubo = tsp_qubo(distance_matrix, n_cities)

# -----------------------------
# Convert QUBO to Ising
# -----------------------------

def qubo_to_ising(Q):
    n = len(Q)
    offset = 0
    pauli_list = []
    for i in range(n):
        for j in range(n):
            if i == j and Q[i][j] != 0:
                z = ['I'] * n
                z[i] = 'Z'
                pauli_list.append(( ''.join(z), 0.5 * Q[i][i] ))
                offset -= 0.5 * Q[i][i]
            elif i < j and Q[i][j] != 0:
                z = ['I'] * n
                z[i] = 'Z'
                z[j] = 'Z'
                pauli_list.append(( ''.join(z), 0.25 * Q[i][j] ))
                offset -= 0.25 * Q[i][j]
    return SparsePauliOp.from_list(pauli_list), offset

ising_hamiltonian, offset = qubo_to_ising(qubo)

# -----------------------------
# QAOA Circuit
# -----------------------------

p = 1  # QAOA layer depth
params = ParameterVector('θ', length=2*p)

def build_qaoa(params):
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for i in range(p):
        gamma = params[2*i]
        beta = params[2*i + 1]

        # Cost unitary
        evo_gate = PauliEvolutionGate(ising_hamiltonian, time=gamma)
        qc.append(evo_gate, range(num_qubits))

        # Mixer unitary
        qc.rx(2 * beta, range(num_qubits))

    return qc

# -----------------------------
# Estimation and Optimization
# -----------------------------

estimator = Estimator()

def energy(theta):
    qc = build_qaoa(params)
    bound_qc = qc.assign_parameters(dict(zip(params, theta)))
    job = estimator.run([ (bound_qc, ising_hamiltonian) ])
    return job.result().values[0].real

initial_theta = np.random.uniform(0, 2*np.pi, 2*p)
res = minimize(energy, initial_theta, method='COBYLA')

# -----------------------------
# Final Result
# -----------------------------

final_theta = res.x
final_circuit = build_qaoa(params).assign_parameters(dict(zip(params, final_theta)))
print("Optimized QAOA circuit:")
print(final_circuit.draw())

print("Optimal energy (approx. tour length):", res.fun + offset)
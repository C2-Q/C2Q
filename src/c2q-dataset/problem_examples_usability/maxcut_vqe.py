from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator

import numpy as np
from scipy.optimize import minimize


def get_maxcut_qubo(n, edges):
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 2
    return Q


def convert_qubo_to_ising(qubo):
    n = len(qubo)
    offset = 0
    pauli_list = []

    for i in range(n):
        for j in range(i, n):
            if i == j:
                coeff = -(1 / 2) * qubo[i][i] - (1 / 4) * np.sum(qubo[i][i+1:]) - (1 / 4) * np.sum(qubo[:, i][:i])
                offset += (1 / 2) * qubo[i][i]
                pauli = ['I'] * n
                pauli[i] = 'Z'
            else:
                coeff = (1 / 4) * qubo[i][j]
                offset += coeff
                pauli = ['I'] * n
                pauli[i] = 'Z'
                pauli[j] = 'Z'
            if coeff != 0:
                pauli_list.append((''.join(pauli), coeff))

    return SparsePauliOp.from_list(pauli_list), offset


def build_vqe_ansatz(n, layers, parameters):
    qc = QuantumCircuit(n)

    # Initial layer of Ry rotations
    for i in range(n):
        qc.ry(parameters[i], i)

    # Entangling CZ layers
    for l in range(layers):
        for i in range(0, n - 1, 2):
            qc.cz(i, i + 1)
        for i in range(1, n - 1, 2):
            qc.cz(i, i + 1)
        # Add parameterized Ry rotations after each layer
        for i in range(n):
            idx = (l + 1) * n + i
            qc.ry(parameters[idx], i)

    qc.measure_all()
    return qc


def cost_function(theta, qc_template, parameters, ising, backend, estimator, eval_log):
    qc_bound = qc_template.assign_parameters(dict(zip(parameters, theta)))
    transpiled = transpile(qc_bound, backend=backend, seed_transpiler=42)
    hamiltonian = ising.apply_layout(transpiled.layout)
    job = estimator.run([(transpiled, hamiltonian)])
    result = job.result()[0].data.evs
    eval_log.append(result)
    return result


def run_vqe_maxcut():
    # 4-node cycle graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n = 4
    layers = 2
    backend = AerSimulator()
    estimator = Estimator(backend)
    estimator.options.default_shots = 1000

    qubo = get_maxcut_qubo(n, edges)
    ising, offset = convert_qubo_to_ising(qubo)

    # Total parameters = (layers + 1) * n
    parameters = ParameterVector('θ', (layers + 1) * n)
    init_theta = np.random.uniform(0, 2 * np.pi, len(parameters))

    # Build ansatz
    qc_template = build_vqe_ansatz(n, layers, parameters)

    # Optimization
    evals = []
    result = minimize(cost_function, init_theta, args=(qc_template, parameters, ising, backend, estimator, evals),
                      method='Powell', options={'maxiter': 300})
    theta_opt = result.x

    # Final execution
    qc_final = build_vqe_ansatz(n, layers, parameters)
    qc_bound = qc_final.assign_parameters(dict(zip(parameters, theta_opt)))
    transpiled = transpile(qc_bound, backend=backend)
    job = backend.run(transpiled, shots=1024)
    counts = job.result().get_counts()

    best_bitstring = max(counts, key=counts.get)
    cut_value = sum(1 for (i, j) in edges if best_bitstring[i] != best_bitstring[j])

    print(f"Best bitstring: {best_bitstring}, Cut size: {cut_value}")
    return counts, best_bitstring


if __name__ == "__main__":
    run_vqe_maxcut()
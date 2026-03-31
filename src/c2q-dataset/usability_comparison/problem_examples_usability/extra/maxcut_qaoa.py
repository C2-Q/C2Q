from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler

import numpy as np
from scipy.optimize import minimize

def get_maxcut_qubo(n, edges):
    """Constructs QUBO matrix for MaxCut."""
    Q = np.zeros((n, n))
    for i, j in edges:
        Q[i, i] -= 1
        Q[j, j] -= 1
        Q[i, j] += 2
    return Q


def convert_qubo_to_ising(qubo):
    n = len(qubo)
    offset = 0
    operator_list = []

    for i in range(n):
        for j in range(i, n):
            if i == j:
                coeff = -(1 / 2) * qubo[i][i] - (1 / 4) * np.sum(qubo[i][(i+1):]) - (1 / 4) * np.sum(qubo[:,i][:i])
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
                operator_list.append((''.join(pauli), coeff))

    return SparsePauliOp.from_list(operator_list), offset


def build_qaoa_circuit(n, ising, parameters, layers):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(layers):
        qc.append(PauliEvolutionGate(ising, parameters[2*i]), range(n))  # cost
        qc.rx(2 * parameters[2*i+1], range(n))  # mixer
    qc.measure_all()
    return qc


def cost_function(theta, qc, parameters, ising, backend, estimator, evals):
    bound_qc = qc.assign_parameters(dict(zip(parameters, theta)))
    transpiled = transpile(bound_qc, backend=backend, seed_transpiler=42)
    ising_applied = ising.apply_layout(transpiled.layout)
    job = estimator.run([(transpiled, ising_applied)])
    result = job.result()[0].data.evs
    evals.append(result)
    return result


def run_qaoa_maxcut():
    # 4-node cycle graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    n = 4
    layers = 1
    backend = AerSimulator()
    estimator = Estimator(backend)
    estimator.options.default_shots = 1000

    qubo = get_maxcut_qubo(n, edges)
    ising, offset = convert_qubo_to_ising(qubo)

    parameters = ParameterVector('θ', 2 * layers)
    initial_theta = [2*np.pi, np.pi] * layers
    qc = build_qaoa_circuit(n, ising, parameters, layers)

    evals = []
    result = minimize(cost_function, initial_theta, args=(qc, parameters, ising, backend, estimator, evals),
                      method='Powell', options={'maxiter': 200})
    theta_opt = result.x

    qc_final = build_qaoa_circuit(n, ising, parameters, layers)
    qc_bound = qc_final.assign_parameters(dict(zip(parameters, theta_opt)))
    transpiled = transpile(qc_bound, backend=backend)
    job = backend.run(transpiled, shots=1024)
    counts = job.result().get_counts()

    best_bitstring = max(counts, key=counts.get)
    cut_value = sum(1 for (i, j) in edges if best_bitstring[i] != best_bitstring[j])

    print(f"Best bitstring: {best_bitstring}, Cut size: {cut_value}")
    return counts, best_bitstring


if __name__ == "__main__":
    run_qaoa_maxcut()
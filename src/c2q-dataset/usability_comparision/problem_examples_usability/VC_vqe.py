import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from scipy.optimize import minimize


def get_mvc_qubo(graph: nx.Graph, A=2.0, B=1.0):
    """Generate the QUBO matrix for Minimum Vertex Cover."""
    n = graph.number_of_nodes()
    Q = np.zeros((n, n))

    for u, v in graph.edges():
        Q[u, u] += -A
        Q[v, v] += -A
        Q[u, v] += A

    for i in range(n):
        Q[i, i] += B

    return Q


def convert_qubo_to_ising(qubo: np.ndarray):
    """Convert QUBO to Ising (Z-basis) Hamiltonian."""
    n = len(qubo)
    pauli_list = []
    offset = 0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                coeff = qubo[i][i] / 2
                offset += qubo[i][i] / 4
                pauli_str = ['I'] * n
                pauli_str[i] = 'Z'
                pauli_list.append(("".join(pauli_str), coeff))
            else:
                coeff = qubo[i][j] / 4
                offset += coeff
                pauli_str = ['I'] * n
                pauli_str[i] = 'Z'
                pauli_str[j] = 'Z'
                pauli_list.append(("".join(pauli_str), coeff))

    return SparsePauliOp.from_list(pauli_list), offset


def build_ansatz(n, layers, parameters):
    """Build a layered Ry + CZ ansatz."""
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(parameters[i], i)
    for l in range(layers):
        for i in range(n - 1):
            qc.cz(i, i + 1)
        for i in range(n):
            idx = (l + 1) * n + i
            qc.ry(parameters[idx], i)
    return qc


def cost_function(theta, qc_template, params, hamiltonian, backend, estimator, log):
    qc = qc_template.assign_parameters(dict(zip(params, theta)))
    qc = transpile(qc, backend)
    h = hamiltonian.apply_layout(qc.layout)
    result = estimator.run([(qc, h)]).result()[0].data.evs
    log.append(result)
    return result


def run_vqe_mvc():
    # Define graph (cycle of 4 nodes)
    G = nx.cycle_graph(4)
    n = G.number_of_nodes()
    layers = 2

    # QUBO → Ising
    qubo = get_mvc_qubo(G)
    ising, offset = convert_qubo_to_ising(qubo)

    # Ansatz setup
    parameters = ParameterVector("θ", (layers + 1) * n)
    qc_template = build_ansatz(n, layers, parameters)
    init_theta = np.random.uniform(0, 2 * np.pi, len(parameters))

    # Backend + Estimator
    backend = AerSimulator()
    estimator = Estimator(backend)
    estimator.options.default_shots = 1024

    # VQE loop
    eval_log = []
    result = minimize(
        cost_function,
        init_theta,
        args=(qc_template, parameters, ising, backend, estimator, eval_log),
        method="Powell",
        options={"maxiter": 200},
    )

    # Final measurement
    qc_final = build_ansatz(n, layers, parameters)
    qc_bound = qc_final.assign_parameters(dict(zip(parameters, result.x)))
    qc_bound.measure_all()
    transpiled = transpile(qc_bound, backend)
    job = backend.run(transpiled, shots=1024)
    counts = job.result().get_counts()

    # Result
    best_bitstring = max(counts, key=counts.get)
    print("Best bitstring:", best_bitstring)

    selected = [i for i, bit in enumerate(reversed(best_bitstring)) if bit == '1']
    print("Selected nodes (cover):", selected)
    return counts, selected


if __name__ == "__main__":
    run_vqe_mvc()
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, StateFn, AerPauliExpectation, CircuitSampler
from qiskit.quantum_info import Pauli
from scipy.optimize import minimize
import numpy as np

# -----------------------------
# Problem setup
# -----------------------------

num_qubits = 4
backend = Aer.get_backend("aer_simulator_statevector")

# Define a sample 4-node graph (as edge list)
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

# -----------------------------
# QAOA Parameters
# -----------------------------

p = 1  # QAOA depth
gamma = Parameter('γ')
beta = Parameter('β')

# -----------------------------
# Build Cost Operator
# -----------------------------

def get_maxcut_hamiltonian(edges, n):
    """Construct MaxCut cost Hamiltonian as PauliSumOp."""
    pauli_list = []
    for (i, j) in edges:
        z_term = ['I'] * n
        z_term[i] = 'Z'
        z_term[j] = 'Z'
        pauli_str = ''.join(z_term)
        pauli_list.append((-0.5) * Pauli(pauli_str))
    return sum(pauli_list)

cost_ham = PauliSumOp.from_list([(op.to_label(), coeff) for op, coeff in [(p, 1.0) for p in get_maxcut_hamiltonian(edges, num_qubits).primitive]])

# -----------------------------
# Build QAOA Circuit
# -----------------------------

def build_qaoa_circuit(gamma_val, beta_val):
    qc = QuantumCircuit(num_qubits)

    # Initial state: |+>^n
    qc.h(range(num_qubits))

    # Apply Cost Unitary
    for (i, j) in edges:
        qc.cx(i, j)
        qc.rz(-2 * gamma_val, j)
        qc.cx(i, j)

    # Apply Mixer Unitary
    for q in range(num_qubits):
        qc.rx(2 * beta_val, q)

    return qc

# -----------------------------
# Expectation Evaluation
# -----------------------------

expectation = AerPauliExpectation()
sampler = CircuitSampler(backend)

def energy_fn(params):
    g, b = params
    qc = build_qaoa_circuit(g, b)
    qc = transpile(qc, backend)
    qc_op = StateFn(cost_ham, is_measurement=True) @ StateFn(qc)
    value = sampler.convert(expectation.convert(qc_op)).eval().real
    return value

# -----------------------------
# Classical Optimization
# -----------------------------

init_params = np.random.rand(2)
res = minimize(energy_fn, init_params, method='COBYLA')

optimal_gamma, optimal_beta = res.x

# -----------------------------
# Final Circuit and Sampling
# -----------------------------

qc_final = build_qaoa_circuit(optimal_gamma, optimal_beta)
qc_final.measure_all()

result = execute(qc_final, backend=Aer.get_backend('qasm_simulator'), shots=1024).result()
counts = result.get_counts()

# -----------------------------
# Interpret Result
# -----------------------------

def bitstring_to_cut(bitstr, edges):
    """Calculate cut size from bitstring."""
    cut = 0
    for (i, j) in edges:
        if bitstr[i] != bitstr[j]:
            cut += 1
    return cut

best_string = max(counts, key=counts.get)
cut_value = bitstring_to_cut(best_string[::-1], edges)  # reverse bitstring for Qiskit ordering

print(f"Best bitstring: {best_string} (cut size = {cut_value})")
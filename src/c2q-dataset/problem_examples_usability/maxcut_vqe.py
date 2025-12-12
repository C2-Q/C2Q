from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.circuit import ParameterVector
from qiskit.opflow import PauliSumOp, StateFn, AerPauliExpectation, CircuitSampler
from qiskit.quantum_info import Pauli
from scipy.optimize import minimize
import numpy as np

# -----------------------------
# Problem setup
# -----------------------------
num_qubits = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
backend = Aer.get_backend("aer_simulator_statevector")

# -----------------------------
# MaxCut Hamiltonian construction
# -----------------------------
def get_maxcut_hamiltonian(n, edge_list):
    terms = []
    for (i, j) in edge_list:
        z = ['I'] * n
        z[i] = 'Z'
        z[j] = 'Z'
        pauli_str = ''.join(z)
        terms.append((pauli_str, 0.5))
    return PauliSumOp.from_list([(s, w) for s, w in terms])

cost_ham = get_maxcut_hamiltonian(num_qubits, edges)

# -----------------------------
# Ansatz construction
# -----------------------------
def create_ansatz(params, num_qubits):
    qc = QuantumCircuit(num_qubits)
    # Initial layer of Ry
    for i in range(num_qubits):
        qc.ry(params[i], i)
    # Entangling layer with CZs
    for (i, j) in edges:
        qc.cz(i, j)
    # Second Ry layer
    for i in range(num_qubits):
        qc.ry(params[num_qubits + i], i)
    return qc

# -----------------------------
# Energy evaluation function
# -----------------------------
expectation = AerPauliExpectation()
sampler = CircuitSampler(backend)

def energy_eval(theta):
    qc = create_ansatz(theta, num_qubits)
    qc = transpile(qc, backend)
    observable = StateFn(cost_ham, is_measurement=True) @ StateFn(qc)
    value = sampler.convert(expectation.convert(observable)).eval()
    return np.real(value)

# -----------------------------
# Run optimization
# -----------------------------
np.random.seed(42)
init_params = np.random.uniform(0, 2*np.pi, 2 * num_qubits)
result = minimize(energy_eval, init_params, method='COBYLA')
optimal_theta = result.x

# -----------------------------
# Execute final circuit
# -----------------------------
qc_final = create_ansatz(optimal_theta, num_qubits)
qc_final.measure_all()
qc_exec = transpile(qc_final, backend=Aer.get_backend("qasm_simulator"))
job = execute(qc_exec, backend=Aer.get_backend("qasm_simulator"), shots=1024)
counts = job.result().get_counts()

# -----------------------------
# Interpretation
# -----------------------------
def bitstring_to_cut(bitstr, edge_list):
    count = 0
    for (i, j) in edge_list:
        if bitstr[i] != bitstr[j]:
            count += 1
    return count

best = max(counts.items(), key=lambda x: x[1])[0][::-1]  # reverse due to endian
cut_value = bitstring_to_cut(best, edges)

print(f"Best cut bitstring: {best}")
print(f"Cut value: {cut_value}")
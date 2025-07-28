"""Quantum Amplitude Estimation for estimating probability of |1>."""

from qiskit import QuantumCircuit, Aer
from qiskit.algorithms import AmplitudeEstimation

# Circuit preparing sqrt(0.2)|1> + sqrt(0.8)|0>
prep = QuantumCircuit(1)
prep.ry(2 * 0.46365, 0)

ae = AmplitudeEstimation(num_eval_qubits=3, quantum_instance=Aer.get_backend('aer_simulator'))
result = ae.estimate(prep)
print('Estimated amplitude:', result.estimation)

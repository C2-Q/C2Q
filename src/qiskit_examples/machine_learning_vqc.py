"""Variational Quantum Classifier on a toy dataset."""

from qiskit import Aer
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit.circuit.library import TwoLocal
import numpy as np

# Toy dataset
training_data = np.array([[0, 1], [1, 0]])
training_labels = np.array([0, 1])

feature_map = RawFeatureVector(2)
ansatz = TwoLocal(2, 'ry', 'cz', reps=1)

vqc = VQC(feature_map, ansatz, training_data, training_labels, quantum_instance=Aer.get_backend('aer_simulator_statevector'))
result = vqc.fit()
print('VQC result:', result)

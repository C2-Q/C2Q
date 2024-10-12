from matplotlib import pyplot as plt

from src.algorithms.grover import grover
from src.circuits_library import quantum_factor_mul_oracle
from src.problems.problem import Problem
#from qiskit_algorithms import AmplificationProblem

from qiskit.circuit.library import PhaseOracle, GroverOperator
#from qiskit_algorithms import AmplificationProblem, Grover
from qiskit import qasm2, QuantumCircuit
from qiskit.primitives import Sampler


# src/algorithms/base_algorithm.py

class Factor(Problem):
    def __init__(self, number):
        self.number = number

    def grover(self, iterations=2):
        oracle, prep_state, obj_bits, working_bits = quantum_factor_mul_oracle(self.number)
        print(list(range(prep_state.num_qubits)))
        grover_circuit = grover(oracle,
                                objective_qubits=obj_bits,
                                iterations=iterations,
                                working_qubits=working_bits,
                                state_pre=prep_state,
                                )
        return grover_circuit


# # Import Qiskit
# from qiskit import QuantumCircuit
# from qiskit.visualization import plot_histogram
# from qiskit import Aer, execute
#
# # Create a function to display the quantum circuit
# def display_circuit(circuit, description):
#     print(description)
#     return circuit.draw(output='mpl')
#
# # AND Gate (Using Toffoli)
# qc_and = QuantumCircuit(3)  # 2 inputs, 1 output
# qc_and.ccx(0, 1, 2)  # Toffoli gate: control on qubits 0 and 1, result on qubit 2
# display_circuit(qc_and, "AND Gate Implementation")
# plt.show()
# # OR Gate (Using Toffoli and X gates)
# qc_or = QuantumCircuit(3)  # 2 inputs, 1 output
# qc_or.x([0, 1])           # Negate inputs
# qc_or.ccx(0, 1, 2)        # Toffoli gate with negated inputs
# qc_or.x([0, 1, 2])        # Negate output and restore original inputs
# display_circuit(qc_or, "OR Gate Implementation")
# plt.show()
# # NOT Gate (Using X gate)
# qc_not = QuantumCircuit(1)  # 1 input, 1 output
# qc_not.x(0)  # NOT gate is just the X gate
# display_circuit(qc_not, "NOT Gate Implementation")
# plt.show()
# # XOR Gate (Using CNOT)
# qc_xor = QuantumCircuit(2)  # 2 inputs
# qc_xor.cx(0, 1)  # CNOT gate: control on qubit 0, result on qubit 1
# display_circuit(qc_xor, "XOR Gate Implementation")
# plt.show()
#
# from qiskit import QuantumCircuit
# from qiskit.visualization import plot_bloch_multivector
# from qiskit.visualization import circuit_drawer
# from qiskit.circuit.library import MCXGate
#
# # Create a quantum circuit with 5 qubits (4 controls and 1 target)
# qc = QuantumCircuit(5)
#
# # Apply Hadamard to the target qubit to turn the MCX into an MCZ
# qc.h(4)  # Target qubit is 4
# # Apply a multi-controlled X gate, which will act as a Z gate due to the Hadamard
# qc.mcx([0, 1, 2, 3], 4)  # Controls are qubits 0, 1, 2, 3
# qc.h(4)  # Apply Hadamard again to return the qubit back
#
# # Draw the circuit
# qc.draw(output='mpl')
# plt.show()
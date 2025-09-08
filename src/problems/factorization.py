import os
from collections import Counter

from matplotlib import pyplot as plt
from pylatex import Figure
from qiskit_aer import AerSimulator

from src.algorithms.grover import grover
from src.circuits_library import quantum_factor_mul_oracle
from src.problems.problem import Problem
#from qiskit_algorithms import AmplificationProblem

from qiskit.circuit.library import PhaseOracle, GroverOperator
#from qiskit_algorithms import AmplificationProblem, Grover
from qiskit import qasm2, QuantumCircuit, transpile
from qiskit.primitives import Sampler


# src/algorithms/base_algorithm.py

class Factor(Problem):
    def __init__(self, number):
        self.number = number

    def grover(self, iterations=2):
        oracle, prep_state, obj_bits, working_bits = quantum_factor_mul_oracle(self.number)
        grover_circuit = grover(oracle,
                                objective_qubits=obj_bits,
                                iterations=iterations,
                                working_qubits=working_bits,
                                state_pre=prep_state,
                                )
        return grover_circuit

    def execute(self):
        qc = self.grover(iterations=2)

        backend = AerSimulator()
        transpiled = transpile(qc, backend)
        shots = 10000

        counts = backend.run(transpiled, shots=shots).result().get_counts()
        return counts

    def interpret(self, result):
        # Get the top 2 most frequent bitstrings
        top_two = Counter(result).most_common(2)

        # Convert bitstrings to integers
        top_two_decimals = [int(bits.replace(' ', ''), 2) for bits, _ in top_two]

        return top_two_decimals

    def report_latex(self, directory=None, output_path=None):
        import time
        import os
        import matplotlib.pyplot as plt
        import networkx as nx
        from pylatex import Document, Section, Subsection, Figure, NoEscape, Package

        if directory is None:
            directory = os.getcwd()

        start_time = time.time()
        problems_name = self.__class__.__name__

        print(f'Starting {problems_name} report generation with LaTeX formatting...')

        # Initialize LaTeX document
        doc = Document()
        doc.packages.append(Package("amsmath"))
        doc.packages.append(Package("qcircuit"))

        with doc.create(Section(f'{problems_name} Problem Report', numbering=False)):
            doc.append(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            doc.append(f'factorization problem: {self.number}')
            with doc.create(Subsection("Grover's Circuit Visualization")):
                try:
                    self._grover_latex(doc, directory)
                except Exception as e:
                    doc.append("not implemented yet\n")

        if output_path is None:
            output_path = os.path.join(directory, f'{problems_name}_report')

        doc.generate_pdf(output_path, compiler="/Library/TeX/texbin/pdflatex", clean_tex=True)
        for img_name in [
            self.grover_circuit_image_path,
        ]:
            print(img_name)
            if img_name is None:
                continue
            img_path = os.path.join(directory, img_name)
            if os.path.exists(img_path):
                os.remove(img_path)

        end_time = time.time()
        print(f"PDF report generated in {end_time - start_time:.2f} seconds.")

    def _grover_latex(self, doc, directory):
        problem_name = self.__class__.__name__
        doc.append(f'The corresponding grover circuit for the {problem_name} is shown below:\n')
        self.grover_circuit_image_path = os.path.join(directory, "quantum_circuit_oracle.png")
        # maximal_independent_set_cnf = maximal_independent_set_to_sat(self.graph)
        grover_qc = self.grover()
        grover_qc.draw(style="mpl")
        plt.savefig(self.grover_circuit_image_path)
        plt.close()
        with doc.create(Figure(position='h!')) as oracle_fig:
            oracle_fig.add_image(self.grover_circuit_image_path, width="300px")
            oracle_fig.add_caption(f'Corresponding Grover Visualization for the {problem_name} Problem')

        from src.algorithms.grover import sample_results
        res = self.execute()
        res = self.interpret(res)
        doc.append("Most Probable Solution for Grover's Algorithm:\n")
        doc.append(f'{self.number} = {res[0]}*{res[1]}')

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

import os
from collections import Counter

from matplotlib import pyplot as plt
from pylatex import Figure
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.problems.basic_arithmetic.arithmetic import Arithmetic
from src.problems.basic_arithmetic.utils import decimal_to_complement_binary_list, complement_binary_list_to_decimal


class Add(Arithmetic):
    def __init__(self, data):
        super().__init__(data)

    def quantum_circuit(self):
        """
        Perform the addition of the two operands (left and right)
        using quantum circuit (measure gates added!!!)
        :return: The result of left + right
                and corresponding circuit
        """
        left = self.left
        right = self.right
        n_bits = max(left.bit_length(), right.bit_length()) + 1
        # Ensure both left and right have n_bits length
        left_list = decimal_to_complement_binary_list(left, n_bits)
        right_list = decimal_to_complement_binary_list(right, n_bits)

        # Create a quantum circuit with 2*n_bits for input and 1 additional for carry
        if left * right > 0:
            qc = QuantumCircuit(3 * n_bits + 1, n_bits + 1)
        else:
            qc = QuantumCircuit(3 * n_bits + 1, n_bits)
        # Initialize the input qubits
        for i in range(n_bits):
            if left_list[i] == 1:
                qc.x(i)  # Set qubit for left bit
            if right_list[i] == 1:
                qc.x(n_bits + i)  # Set qubit for right bit

        # Apply quantum gates for addition
        for i in range(n_bits):
            qc.ccx(i, n_bits + i, 2 * n_bits + i + 1)
            qc.cx(i, n_bits + i)
            qc.ccx(n_bits + i, 2 * n_bits + i, 2 * n_bits + i + 1)
            qc.cx(n_bits + i, 2 * n_bits + i)
            qc.cx(i, n_bits + i)

        # Measuring the result
        for i in range(n_bits):
            qc.measure(i + 2 * n_bits, i)  # Measure left bits to result bits
        if left * right > 0:
            qc.measure(3 * n_bits, n_bits)
        # execution required from recommender side
        # sampler = Sampler()
        # result = sampler.run(circuits=qc, shots=1024).result()
        # result = list(result.quasi_dists[0].keys())[0]
        # if left * right > 0:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        # else:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return qc

    # def execute(self):
    #     qc = self.quantum_circuit()
    #
    #     backend = AerSimulator()
    #     transpiled = transpile(qc, backend)
    #     shots = 10000
    #
    #     counts = backend.run(transpiled, shots=shots).result().get_counts()
    #     return counts
    #
    def interpret(self, result):
        result = Counter(result).most_common(1)
        result = result[0][0]

        result = int(result.replace(' ', ''), 2)

        n_bits = max(self.left.bit_length(), self.right.bit_length()) + 1
        if self.left * self.right > 0:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        else:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return result
    #
    # def report_latex(self, directory=None):
    #     import time
    #     import os
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     from pylatex import Document, Section, Subsection, Figure, NoEscape, Package
    #
    #     if directory is None:
    #         directory = os.getcwd()
    #
    #     start_time = time.time()
    #     problems_name = self.__class__.__name__
    #
    #     print(f'Starting {problems_name} report generation with LaTeX formatting...')
    #
    #     # Initialize LaTeX document
    #     doc = Document()
    #     doc.packages.append(Package("amsmath"))
    #     doc.packages.append(Package("qcircuit"))
    #
    #     with doc.create(Section(f'{problems_name} Problem Report', numbering=False)):
    #         doc.append(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    #         doc.append(f'addition problem: {self.left} + {self.right}')
    #         with doc.create(Subsection("Quantum Circuit Visualization")):
    #             # try:
    #             #     self._qc_latex(doc, directory)
    #             # except Exception as e:
    #             #     print(e.__str__())
    #             #     doc.append("not implemented yet\n")
    #             self._qc_latex(doc, directory=directory)
    #
    #     output_path = os.path.join(directory, f'{problems_name}_report')
    #
    #     doc.generate_pdf(output_path, compiler="/Library/TeX/texbin/pdflatex", clean_tex=True)
    #     for img_name in [
    #         self.qc_image_path,
    #     ]:
    #         print(img_name)
    #         if img_name is None:
    #             continue
    #         img_path = os.path.join(directory, img_name)
    #         if os.path.exists(img_path):
    #             os.remove(img_path)
    #
    #     end_time = time.time()
    #     print(f"PDF report generated in {end_time - start_time:.2f} seconds.")
    #
    # def _qc_latex(self, doc, directory):
    #     problem_name = self.__class__.__name__
    #     doc.append(f'The corresponding quantum circuit for the {problem_name} is shown below:\n')
    #     self.qc_image_path = os.path.join(directory, "quantum_circuit.png")
    #     qc = self.quantum_circuit()
    #     qc.draw(style="mpl")
    #     plt.savefig(self.qc_image_path)
    #     plt.close()
    #     with doc.create(Figure(position='h!')) as oracle_fig:
    #         oracle_fig.add_image(self.qc_image_path, width="300px")
    #         oracle_fig.add_caption(f'Corresponding Quantum Circuit Visualization for the {problem_name} Problem')
    #
    #     from src.algorithms.grover import sample_results
    #     res = self.execute()
    #     res = self.interpret(res)
    #     doc.append("Most Probable Solution for the Algorithm:\n")
    #     doc.append(f'{self.left} + {self.right}  = {res}')

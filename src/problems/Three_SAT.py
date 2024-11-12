import os
import time

import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict
from fpdf import FPDF
from qiskit import QuantumCircuit

from src.algorithms.QAOA.QAOA import qaoa_no_optimization, qaoa_optimize, sample_results
from src.algorithms.VQE.VQE import vqe_optimization
from src.algorithms.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt

from src.recommender.recommender_engine import recommender
from pysat.formula import CNF

from src.reduction import solve_all_cnf_solutions
from src.sat_to_qubo import Chancellor


class ThreeSat(NP):
    """
    An application class for the clique problem based on a NetworkX graph.
    """

    def __init__(self, formula: CNF) -> None:
        super().__init__()
        # add a code to check each clause <= 3
        self.three_sat = formula
        self.cha = Chancellor(self.three_sat)

    def to_qubo(self) -> 'QUBO':
        self.cha.fillQ()
        num_vars = max(max(x, y) for x, y in self.cha.Q.keys()) + 1
        Q = np.zeros((num_vars, num_vars))

        # Iterate over self.cha.Q and populate the Q matrix
        for (i, j), value in self.cha.Q.items():
            Q[i, j] = value

        return QUBO(Q)

    def grover_sat(self, iterations=1):
        oracle = cnf_to_quantum_oracle_optimized(self.three_sat)
        state_prep = QuantumCircuit(oracle.num_qubits)
        state_prep.h(list(range(self.three_sat.nv)))
        grover_circuit = grover(oracle=oracle,
                                objective_qubits=list(range(self.three_sat.nv)),
                                working_qubits=list(range(self.three_sat.nv)),
                                state_pre=state_prep,
                                iterations=iterations)
        print(grover_circuit)
        return grover_circuit

    def report(self) -> None:
        """
        Generate a PDF report for solving the 3-SAT problem using QAOA, VQE, and Grover's algorithm.
        The report includes the 3-SAT formula, the QUBO matrix visualization, algorithm results, and timestamps.
        """
        # Start timer
        start_time = time.time()
        print("Starting 3-SAT problem report generation...")

        # Convert 3-SAT to QUBO and visualize
        Q = self.to_qubo()
        qubo = Q.Q
        qubo_matrix = Q.display_matrix()  # Get QUBO matrix as a NumPy array
        classical_solutions = solve_all_cnf_solutions(self.three_sat)
        # Track time for QUBO generation and visualization
        qubo_time = time.time()
        print(f"QUBO matrix generated and visualized in {qubo_time - start_time:.2f} seconds.")

        # Initialize PDF for the report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", "B", 16)
        pdf.cell(0, 10, "3-SAT Solution Report for Problem Reduced from Original Format", ln=True, align="C")

        # Add timestamp for report generation start
        pdf.set_font("Times", "I", 10)
        pdf.cell(0, 10, f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
                 ln=True)

        # Add 3-SAT formula to the report
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "3-SAT Formula:", ln=True)
        pdf.set_font("Times", "", 12)
        formula_str = str(self.three_sat.clauses)
        pdf.multi_cell(0, 10, formula_str)

        # Add classical solutions to the report
        pdf.set_font("Times", "B", 14)
        pdf.ln(5)
        pdf.cell(0, 10, "Exact Solutions:", ln=True)
        pdf.set_font("Times", "", 12)
        classical_solutions_str = "\n".join(str(solution) for solution in classical_solutions)
        pdf.multi_cell(0, 10, classical_solutions_str)

        # Add QUBO matrix visualization to the report
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Converted QUBO Matrix Visualization", ln=True)
        pdf.set_font("Times", "", 12)
        num_rows = max(max(x, y) for x, y in self.cha.Q.keys()) + 1
        num_ansila = num_rows - self.three_sat.nv
        pdf.cell(0, 10, f"It is a {num_rows} × {num_rows} upper matrix with {num_ansila} ansilla variables.", ln=True)
        formatted_qubo = "[\n" + "\n".join(
            ["  [" + " ".join(f"{val:.1f}" for val in row) + "]" for row in qubo_matrix]) + "\n]"
        pdf.multi_cell(0, 10, formatted_qubo)

        if os.path.exists("qubo_matrix.png"):
            pdf.image("qubo_matrix.png", x=10, y=30, w=180)  # Adjust as needed

        # Add oracle image here
        pdf.add_page()
        oracle_circuit_image_path = "quantum_circuit_oracle.png"
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Corresponding Oracle Visualization", ln=True)
        pdf.set_y(pdf.get_y() - 5)
        pdf.set_font("Times", "", 12)
        oracle = cnf_to_quantum_oracle_optimized(self.three_sat)
        oracle.decompose().draw(style="mpl")
        plt.savefig(oracle_circuit_image_path)
        plt.close()
        pdf.image(oracle_circuit_image_path, x=10, y=pdf.get_y(), w=190)
        # QAOA Solution
        from src.algorithms.QAOA.QAOA import sample_results
        qaoa_dict = qaoa_optimize(qubo, layers=3)
        qaoa_solution = sample_results(qaoa_dict["qc"], qaoa_dict["parameters"], qaoa_dict["theta"])
        print(qaoa_dict["parameters"])
        print(qaoa_dict["theta"])
        # Add QAOA Solution and Configurations to the report
        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "QAOA Solution:", ln=True)
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(0, 10, f"QAOA Highest Probable Solution: {qaoa_solution}")

        # Add some space before QAOA Configurations
        pdf.ln(5)

        # Add QAOA Configurations with bullet points, within a specific width to avoid overflow
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "QAOA Configurations:", ln=True)
        pdf.set_font("Times", "", 12)
        qaoa_config = [
            "- Layers: 3",
            "- Maximizer Hamiltonian: Standard Mixing Hamiltonian",
            "- Classical Optimizer: Powell's Method",
            "- Maximum Iterations: 500",
            "- Initialization:",
            "   - Gamma: 2pi",
            "   - Beta: pi",
        ]

        pdf.multi_cell(0, 10, "\n".join(qaoa_config), border=0, align='L', max_line_height=pdf.font_size)

        # Add space before the next section if needed
        pdf.ln(5)

        # QAOA Circuit start here !!!
        qaoa_circuit_image_path = "quantum_circuit_qaoa.png"
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(5)

        # Plot and save the quantum circuit for qaoa !!
        qaoa_dict["qc"].decompose().draw(style="mpl")
        plt.savefig(qaoa_circuit_image_path)
        plt.close()

        # Insert the qaoa quantum circuit image into the PDF
        pdf.image(qaoa_circuit_image_path, x=10, y=pdf.get_y(), w=190)
        # QAOA circuits layout ends here !!!

        # VQE Solution
        from src.algorithms.VQE.VQE import sample_results
        vqe_dict = vqe_optimization(qubo, layers=3)
        vqe_solution = sample_results(vqe_dict["qc"], vqe_dict["parameters"], vqe_dict["theta"])

        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "VQE Solution:", ln=True)
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(0, 10, f"VQE Highest Probable Solution: {vqe_solution}")

        # Add some space before QAOA Configurations
        pdf.ln(5)

        # Add QAOA Configurations with bullet points, within a specific width to avoid overflow
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "VQE Configurations:", ln=True)
        pdf.set_font("Times", "", 12)
        qaoa_config = [
            "- Layers: 3",
            "- Ansatz: Two Local",
            "- Classical Optimizer: Powell's Method",
            "- Maximum Iterations: 500",
            "- Initialization:",
            "   - Theta: pi",
        ]

        # Specify the width of the multi_cell to prevent overflow
        pdf.multi_cell(0, 10, "\n".join(qaoa_config), border=0, align='L', max_line_height=pdf.font_size)

        # Add space before the next section if needed
        pdf.ln(5)

        # VQE Circuit start here !!!
        vqe_circuit_image_path = "quantum_circuit_vqe.png"
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "VQE Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(5)

        # Plot and save the quantum circuit for qaoa !!
        vqe_dict["qc"].decompose().draw(style="mpl")
        plt.savefig(vqe_circuit_image_path)
        plt.close()

        # Insert the qaoa quantum circuit image into the PDF
        pdf.image(vqe_circuit_image_path, x=10, y=pdf.get_y(), w=190)
        # VQE circuits layout ends here !!!

        # Grover's Algorithm Solution
        grover_circuit = self.grover_sat(iterations=1)
        from src.algorithms.grover import sample_results
        grover_solution = sample_results(grover_circuit)

        pdf.add_page()
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Grover's Algorithm Solution:", ln=True)
        pdf.set_font("Times", "", 12)
        pdf.multi_cell(0, 10, f"Grover's Most Probable Solution: {grover_solution}")

        # Grover Circuit start here !!!
        grover_circuit_image_path = "quantum_circuit_grover.png"
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "Grover Search, generated quantum circuit", ln=True, align='C')
        pdf.ln(5)

        # Plot and save the quantum circuit for qaoa !!
        grover_circuit.draw(style="mpl")
        plt.savefig(grover_circuit_image_path)
        plt.close()

        # Insert the qaoa quantum circuit image into the PDF
        pdf.image(grover_circuit_image_path, x=10, y=pdf.get_y(), w=190)
        # Grover circuits layout ends here !!!

        # End of report, save PDF
        output_path = "ThreeSat_Solution_Report.pdf"
        pdf.output(output_path)

        # Final timestamp and total time
        end_time = time.time()
        print(f"Report generation completed in {end_time - start_time:.2f} seconds.")
        print(f"PDF report generated and saved to {output_path}")

        if os.path.exists(qaoa_circuit_image_path):
            os.remove(qaoa_circuit_image_path)
        if os.path.exists(oracle_circuit_image_path):
            os.remove(oracle_circuit_image_path)
        if os.path.exists(vqe_circuit_image_path):
            os.remove(vqe_circuit_image_path)
        if os.path.exists(grover_circuit_image_path):
            os.remove(grover_circuit_image_path)
import os
import time

import networkx
import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from fpdf import FPDF
from pylatex import Document, Package, Section, Subsection, Figure, NoEscape
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from src.algorithms.QAOA.QAOA import qaoa_optimize
from src.algorithms.VQE.VQE import vqe_optimization
from src.algorithms.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized
from src.graph import Graph
from src.problems.Three_SAT import ThreeSat
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt

from src.recommender.recommender_engine import recommender
from src.reduction import independent_set_to_sat, maximal_independent_set_to_sat, sat_to_3sat
from src.problems.np_complete import NPC


class MIS(NPC):
    """
    An application class for the maximal independent set problem based on a NetworkX graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """
        Args:
            graph: A graph representing the problem. It can be specified directly as a
                   NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
            size: The desired size of the clique (K).
        """
        # If the graph is not a NetworkX graph, convert it
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, Graph):
            self.graph = graph.G
        else:
            raise TypeError("The graph must be a NetworkX graph")

        # Store nodes and mappings
        self.nodes = list(self.graph.nodes())
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def reduce_to_sat(self):
        n = len(self.nodes)
        self.sat = independent_set_to_sat(self.graph)

    def to_qubo(self, A: float = 1.0, B: float = 1.0) -> 'QUBO':
        """
        Converts the mis problem into a QUBO problem represented by a QUBO class instance
        based on the Hamiltonian H = H_A + H_B
        Was done intuitively...
         H_A = A*sum_((u,v)\in E)(x_u*x_v)
         H_B = -B*sum_x_v
        Args:
            A: Penalty weight
            B: Penalty weight.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        n = len(self.nodes)
        Q = np.zeros((n, n))
        A = 2 * B

        # Add linear terms to Q diagonal
        for idx in range(n):
            Q[idx, idx] -= B

        # Add quadratic terms (upper triangular part only)
        for i in range(n):
            for j in range(i + 1, n):
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                if self.graph.has_edge(node_i, node_j):
                    Q[i, j] += A

        return QUBO(Q)

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of node indices forming the clique.

        Args:
            result: The calculated result of the problem (binary vector).

        Returns:
            The list of node indices whose corresponding variable is 1.
        """
        x = np.array(result)
        nodes_in_mis = []
        for idx, val in enumerate(x):
            if val == 1:
                node_label = self.indices_node[idx]
                nodes_in_mis.append(node_label)
        return nodes_in_mis

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with nodes in the independent set highlighted.
        Args:
            result: The calculated result for the problem (binary vector).
            pos: The positions of nodes (optional).
        """

        x = np.array(result)
        node_colors = {}
        for idx, val in enumerate(x):
            node_label = self.indices_node[idx]
            if val == 1:
                node_colors[node_label] = 'red'  # Nodes in the clique are red
            else:
                node_colors[node_label] = 'gray'  # Other nodes are gray

        graph_nodes = list(self.graph.nodes())
        color_map = [node_colors[node] for node in graph_nodes]

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.graph,
            node_color=color_map,
            pos=pos,
            with_labels=True,
            node_size=500,
            font_size=12,
            font_color='white',
            edge_color='black'
        )
        # plt.show()

    def to_sat(self):
        self.sat = maximal_independent_set_to_sat(self.graph)

    def grover_sat(self, iterations=1):
        independent_set_cnf = independent_set_to_sat(self.graph)
        maximal_independent_set_cnf = maximal_independent_set_to_sat(self.graph)
        oracle = cnf_to_quantum_oracle_optimized(maximal_independent_set_cnf)
        state_prep = QuantumCircuit(oracle.num_qubits)
        state_prep.h(list(range(self.graph.number_of_nodes())))
        grover_circuit = grover(oracle=oracle,
                                objective_qubits=list(range(self.graph.number_of_nodes())),
                                working_qubits=list(range(self.graph.number_of_nodes())),
                                state_pre=state_prep,
                                iterations=iterations)
        return grover_circuit

    def report_3sat(self):
        formula = maximal_independent_set_to_sat(self.graph)
        formula = sat_to_3sat(formula)
        sat = ThreeSat(formula)
        sat.report_latex()

    def report(self) -> None:
        """
        Generates a PDF report summarizing the problem, its solution, and a visualization of the result.
        """
        start_time = time.time()
        image_path = "graph_visualization.png"
        qaoa_circuit_image_path = "quantum_circuit_qaoa.png"
        qaoa_solution_image_path = "qaoa_solution_visualization.png"
        vqe_circuit_image_path = "vqe_quantum_circuit_qaoa.png"
        vqe_solution_image_path = "vqe_solution_visualization.png"
        grover_circuit_image_path = "grover_circuit.png"
        grover_solution_image_path = "grover_solution_visualization.png"

        pdf = FPDF()
        pdf.set_font("Times", size=12)
        # New page
        pdf.add_page()

        # Set title with Times New Roman font
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "MIS Problem Report", ln=True, align='C')

        # Add some details about the graph
        pdf.set_font("Times", size=12)
        pdf.ln(10)  # Add some vertical space
        pdf.cell(200, 10, f"GRAPH:", ln=True, align='L')
        pdf.cell(200, 10, f"Number of Nodes: {len(self.nodes)}", ln=True, align='L')

        # Display edges in a single line
        edges = list(self.graph.edges())
        edge_str = ', '.join([f"({u},{v})" for u, v in edges])
        pdf.cell(200, 10, f"Edges of Nodes: [{edge_str}]", ln=True, align='L')

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
        plt.savefig(image_path)
        plt.close()

        # Insert the image into the PDF
        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of Graph:", ln=True, align='L')
        pdf.image(image_path, x=10, y=pdf.get_y(), w=190)

        # Perform QUBO optimization and sampling using QAOA
        qubo = self.to_qubo().Q
        qaoa_dict = qaoa_optimize(qubo, layers=3)
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]
        # recommender(qc)

        # Sample the QAOA circuit and get the most probable solution
        from src.algorithms.QAOA.QAOA import sample_results
        highest_possible_solution = sample_results(qc, parameters, theta)

        # for qaoa
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for qaoa !!
        qc.decompose().draw(style="mpl")
        plt.savefig(qaoa_circuit_image_path)
        plt.close()

        # Insert the qaoa quantum circuit image into the PDF
        pdf.image(qaoa_circuit_image_path, x=10, y=pdf.get_y(), w=190)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization Results", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Times", size=12)
        pdf.cell(200, 10, "Most Probable Solution:", ln=True, align='L')
        pdf.cell(200, 10, f"{highest_possible_solution}", ln=True, align='L')

        plt.figure(figsize=(8, 6))
        self.draw_result(highest_possible_solution, pos=pos)  # Reuse the graph positions

        plt.savefig(qaoa_solution_image_path)
        plt.close()

        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of QAOA Solution:", ln=True, align='L')
        pdf.image(qaoa_solution_image_path, x=10, y=pdf.get_y(), w=190)

        #pdf_output_path = "maxcut_report.pdf"
        #pdf.output(pdf_output_path)

        # start here for vqe algorithm
        # Perform QUBO optimization and sampling using VQE
        qubo = self.to_qubo().Q
        vqe_dict = vqe_optimization(qubo, layers=1)
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]
        # recommender(qc)

        # Sample the vqe circuit and get the most probable solution
        from src.algorithms.VQE.VQE import sample_results
        highest_possible_solution = sample_results(qc, parameters, theta)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "VQE Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for qaoa !!
        qc.decompose().draw(style="mpl")
        plt.savefig(vqe_circuit_image_path)
        plt.close()

        # Insert the quantum circuit image into the PDF
        pdf.image(vqe_circuit_image_path, x=10, y=pdf.get_y(), w=190)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "VQE Optimization Results", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Times", size=12)
        pdf.cell(200, 10, "Most Probable Solution:", ln=True, align='L')
        pdf.cell(200, 10, f"{highest_possible_solution}", ln=True, align='L')

        plt.figure(figsize=(8, 6))
        self.draw_result(highest_possible_solution, pos=pos)  # Reuse the graph positions
        plt.savefig(vqe_solution_image_path)
        plt.close()

        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of VQE Solution:", ln=True, align='L')
        pdf.image(vqe_solution_image_path, x=10, y=pdf.get_y(), w=190)

        # oracle execution and visualization, pdf written

        grover_circuit = self.grover_sat(iterations=1)
        from src.algorithms.grover import sample_results
        most_probable_grover_result = sample_results(grover_circuit)
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "Grover algorithm, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for grover !!
        grover_circuit.draw(style="mpl")
        plt.savefig(grover_circuit_image_path)
        plt.close()
        # Insert the quantum circuit image into the PDF
        pdf.image(grover_circuit_image_path, x=10, y=pdf.get_y(), w=190)
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "Grover Optimization Results", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Times", size=12)
        pdf.cell(200, 10, "Most Probable Solution:", ln=True, align='L')
        pdf.cell(200, 10, f"{most_probable_grover_result}", ln=True, align='L')

        plt.figure(figsize=(8, 6))
        self.draw_result(most_probable_grover_result, pos=pos)  # Reuse the graph positions
        plt.savefig(grover_solution_image_path)
        plt.close()

        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of Grover Solution:", ln=True, align='L')
        pdf.image(grover_solution_image_path, x=10, y=pdf.get_y(), w=190)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        recommender_output, recommender_devices = recommender(qc)
        pdf.cell(200, 10, "Devices recommendation based on VQE circuit", ln=True, align='L')

        # Use multi_cell to handle long text
        pdf.set_font("Times", '', 12)  # Optionally set a smaller font for the output text
        pdf.multi_cell(0, 10, recommender_output, align='L')

        pdf_output_path = "independent_set_report.pdf"
        pdf.output(pdf_output_path)
        # clean up the saved PNG images

        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(qaoa_circuit_image_path):
            os.remove(qaoa_circuit_image_path)
        if os.path.exists(qaoa_solution_image_path):
            os.remove(qaoa_solution_image_path)
        if os.path.exists(vqe_circuit_image_path):
            os.remove(vqe_circuit_image_path)
        if os.path.exists(vqe_solution_image_path):
            os.remove(vqe_solution_image_path)
        if os.path.exists(grover_circuit_image_path):
            os.remove(grover_circuit_image_path)
        if os.path.exists(grover_solution_image_path):
            os.remove(grover_solution_image_path)

        print(f"PDF report saved as {pdf_output_path}")

    # def report_latex(self, directory: str = None):
    #     import time
    #     import os
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     from pylatex import Document, Section, Subsection, Figure, NoEscape, Package
    #
    #     if directory is None:
    #         directory = os.getcwd()
    #     start_time = time.time()
    #     print("Starting Independent Set problem report generation with LaTeX formatting...")
    #
    #     # Initialize LaTeX document
    #     doc = Document()
    #     doc.packages.append(Package("amsmath"))
    #     doc.packages.append(Package("qcircuit"))
    #
    #     # Compute layout once
    #     pos = nx.spring_layout(self.graph)
    #
    #     with doc.create(Section("Independent Set Problem Report", numbering=False)):
    #         doc.append(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    #
    #         # Graph details
    #         with doc.create(Subsection("Graph Details", numbering=False)):
    #             doc.append(f"Number of Nodes: {len(self.nodes)}\n")
    #             edges = list(self.graph.edges())
    #             edge_str = ', '.join([f"({u},{v})" for u, v in edges])
    #             doc.append(f"Edges of Nodes: [{edge_str}]\n")
    #
    #             plt.figure(figsize=(8, 6))
    #             nx.draw(self.graph, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray')
    #             graph_image_path = os.path.join(directory, "graph_visualization.png")
    #             plt.title("Independent Set Graph")
    #             plt.savefig(graph_image_path)
    #             plt.close()
    #
    #         with doc.create(Figure(position='h!')) as graph_fig:
    #             graph_fig.add_image(graph_image_path, width="360px")
    #             graph_fig.add_caption("Graph Visualization")
    #
    #         # QUBO Matrix Visualization
    #         with doc.create(Subsection('QUBO Matrix Visualization')):
    #             doc.append("Converted QUBO matrix visualization:\n")
    #             qubo_matrix = self.to_qubo().Q
    #             matrix_latex = "$\\begin{bmatrix}\n" + \
    #                            "\\\\".join(" & ".join(f"{val:.1f}" for val in row) for row in qubo_matrix) + \
    #                            "\n\\end{bmatrix}$"
    #             doc.append(NoEscape(matrix_latex))
    #
    #         # Oracle Visualization
    #         with doc.create(Subsection("Oracle Visualization")):
    #             doc.append("The corresponding oracle for the Independent Set problem is shown below:\n")
    #             oracle_circuit_image_path = os.path.join(directory, "quantum_circuit_oracle.png")
    #             maximal_independent_set_cnf = maximal_independent_set_to_sat(self.graph)
    #             oracle = cnf_to_quantum_oracle_optimized(maximal_independent_set_cnf)
    #             oracle.draw(style="mpl")
    #             plt.savefig(oracle_circuit_image_path)
    #             plt.close()
    #
    #             with doc.create(Figure(position='h!')) as oracle_fig:
    #                 oracle_fig.add_image(oracle_circuit_image_path, width="300px")
    #                 oracle_fig.add_caption("Corresponding Oracle Visualization for the Independent Set Problem")
    #
    #         # QAOA Section
    #         with doc.create(Subsection("QAOA Optimization Results", numbering=False)):
    #             qubo = self.to_qubo().Q
    #             qaoa_dict = qaoa_optimize(qubo, layers=3)
    #             qaoa_qc = qaoa_dict["qc"]
    #             parameters = qaoa_dict["parameters"]
    #             theta = qaoa_dict["theta"]
    #             from src.algorithms.QAOA.QAOA import sample_results
    #             qaoa_solution = sample_results(qaoa_qc, parameters, theta)
    #             doc.append("Most Probable Solution for QAOA:\n")
    #             doc.append(NoEscape(r"\begin{itemize}"))
    #             for i, state in enumerate(qaoa_solution):
    #                 assignment = "true" if state else "false"
    #                 doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
    #             doc.append(NoEscape(r"\end{itemize}"))
    #
    #             self.draw_result(qaoa_solution, pos=pos)
    #             qaoa_result_image_path = os.path.join(directory, "qaoa_result.png")
    #             plt.savefig(qaoa_result_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as qaoa_res_fig:
    #                 qaoa_res_fig.add_image(qaoa_result_image_path, width="180px")
    #                 qaoa_res_fig.add_caption("QAOA Result")
    #
    #             qaoa_circuit_image_path = os.path.join(directory, "quantum_circuit_qaoa.png")
    #             qaoa_qc.decompose().draw(style="mpl")
    #             plt.savefig(qaoa_circuit_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as qaoa_fig:
    #                 qaoa_fig.add_image(qaoa_circuit_image_path, width="180px")
    #                 qaoa_fig.add_caption("QAOA Quantum Circuit")
    #
    #         # VQE Section
    #         with doc.create(Subsection("VQE Optimization Results", numbering=False)):
    #             start_time = time.perf_counter()
    #             vqe_dict = vqe_optimization(qubo, layers=3)
    #             end_time = time.perf_counter()
    #             execution_time = end_time - start_time
    #             print(f"Circuit execution time: {execution_time:.4f} seconds")
    #             vqe_qc = vqe_dict["qc"]
    #             print(vqe_dict["qc"].decompose().decompose().decompose().depth())
    #             parameters = vqe_dict["parameters"]
    #             theta = vqe_dict["theta"]
    #             from src.algorithms.VQE.VQE import sample_results
    #             vqe_solution = sample_results(vqe_qc, parameters, theta)
    #             doc.append("Most Probable Solution for VQE:\n")
    #             doc.append(NoEscape(r"\begin{itemize}"))
    #             for i, state in enumerate(vqe_solution):
    #                 assignment = "true" if state else "false"
    #                 doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
    #             doc.append(NoEscape(r"\end{itemize}"))
    #
    #             self.draw_result(vqe_solution, pos=pos)
    #             vqe_result_image_path = os.path.join(directory, "vqe_result.png")
    #             plt.savefig(vqe_result_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as vqe_res_fig:
    #                 vqe_res_fig.add_image(vqe_result_image_path, width="180px")
    #                 vqe_res_fig.add_caption("VQE Result")
    #
    #             vqe_circuit_image_path = os.path.join(directory, "quantum_circuit_vqe.png")
    #             vqe_qc.decompose().draw(style="mpl")
    #             plt.savefig(vqe_circuit_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as vqe_fig:
    #                 vqe_fig.add_image(vqe_circuit_image_path, width="180px")
    #                 vqe_fig.add_caption("VQE Quantum Circuit")
    #
    #         # Grover Section
    #         with doc.create(Subsection("Grover's Algorithm Results", numbering=False)):
    #             grover_qc = self.grover_sat(iterations=1)
    #             from src.algorithms.grover import sample_results
    #             grover_solution = sample_results(grover_qc)
    #             doc.append("Most Probable Solution for Grover's Algorithm:\n")
    #             doc.append(NoEscape(r"\begin{itemize}"))
    #             for i, state in enumerate(grover_solution):
    #                 assignment = "true" if state else "false"
    #                 doc.append(NoEscape(rf"\item Variable \( x_{{{i + 1}}} \) is set to {assignment}"))
    #             doc.append(NoEscape(r"\end{itemize}"))
    #             # print(grover_solution)
    #             self.draw_result(grover_solution, pos=pos)
    #             grover_result_image_path = os.path.join(directory, "grover_result.png")
    #             plt.savefig(grover_result_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as grover_res_fig:
    #                 grover_res_fig.add_image(grover_result_image_path, width="180px")
    #                 grover_res_fig.add_caption("Grover's Algorithm Result")
    #
    #             grover_circuit_image_path = os.path.join(directory, "quantum_circuit_grover.png")
    #             grover_qc.draw(style="mpl")
    #             plt.savefig(grover_circuit_image_path)
    #             plt.close()
    #             with doc.create(Figure(position='h!')) as grover_fig:
    #                 grover_fig.add_image(grover_circuit_image_path, width="180px")
    #                 grover_fig.add_caption("Grover's Quantum Circuit")
    #
    #         # Optional: recommend device
    #         string, _ = recommender(qaoa_qc, save_figures=True)
    #         # Insert recommender plots into report
    #         for plot_name, caption in zip([
    #             "recommender_errors_devices.png",
    #             "recommender_times_devices.png",
    #             "recommender_prices_devices.png"
    #         ], [
    #             "Estimated total error with each quantum computer",
    #             "Estimated total time with each quantum computer",
    #             "Estimated price with each quantum computer"
    #         ]):
    #             fig_path = os.path.join(directory, plot_name)
    #             if os.path.exists(fig_path):
    #                 with doc.create(Figure(position='h!')) as fig:
    #                     fig.add_image(fig_path, width="360px")
    #                     fig.add_caption(caption)
    #         # error_image_path = "error_image.png"
    #         # price_image_path = "price_image.png"
    #         # time_image_path = "time_image.png"
    #         # plt.savefig(error_image_path)
    #         # plt.savefig(price_image_path)
    #         # plt.savefig(time_image_path)
    #         # plt.close()
    #         # print(string)
    #
    #         with doc.create(Subsection("Device Recommendation Summary", numbering=False)):
    #             doc.append("Here is the device recommendation summary based on error, time, and price:\\\n")
    #             doc.append(string)
    #
    #     output_path = os.path.join(directory, "independent_set_report_with_latex")
    #     # output_path = "independent_set_report_with_latex.pdf"
    #     doc.generate_pdf(output_path, compiler="/Library/TeX/texbin/pdflatex", clean_tex=True)
    #
    #     # Cleanup temporary images
    #     for img_name in [
    #         graph_image_path, qaoa_result_image_path, qaoa_circuit_image_path,
    #         vqe_result_image_path, vqe_circuit_image_path,
    #         grover_result_image_path, grover_circuit_image_path,
    #         oracle_circuit_image_path
    #     ]:
    #         img_path = os.path.join(directory, img_name)
    #         if os.path.exists(img_path):
    #             os.remove(img_path)
    #
    #     end_time = time.time()
    #     print(f"PDF report generated in {end_time - start_time:.2f} seconds.")

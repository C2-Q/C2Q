import os

import networkx
import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from fpdf import FPDF
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from src.algorithms.QAOA.QAOA import qaoa_optimize
from src.algorithms.VQE.VQE import vqe_optimization
from src.algorithms.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt

from src.recommender.recommender_engine import recommender
from src.reduction import independent_set_to_sat, maximal_independent_set_to_sat


class MIS(NP):
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
        # Create a mapping from node labels to their corresponding x values
        node_colors = {}
        for idx, val in enumerate(x):
            node_label = self.indices_node[idx]
            if val == 1:
                node_colors[node_label] = 'red'  # Nodes in the clique are red
            else:
                node_colors[node_label] = 'gray'  # Other nodes are gray

        # Get the nodes in the order that nx.draw will use
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

    def report(self, backend=AerSimulator()) -> None:
        """
        Generates a PDF report summarizing the problem, its solution, and a visualization of the result.
        """
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
        pdf.cell(200, 10, "MaxCut Problem Report, Solved on Helmi", ln=True, align='C')

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
        qaoa_dict = qaoa_optimize(qubo, layers=1, backend=backend)
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]
        # recommender(qc)

        # Sample the QAOA circuit and get the most probable solution
        from src.algorithms.QAOA.QAOA import sample_results
        highest_possible_solution = sample_results(qc, parameters, theta, backend)

        # Add a new page for QAOA results
        # Draw and insert the quantum circuit (qc) into the PDF
        # for qaoa
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for qaoa !!
        qc.decompose().draw("mpl", filename=qaoa_circuit_image_path)

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
        vqe_dict = vqe_optimization(qubo, layers=1, backend=backend)
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]
        # recommender(qc)

        # Sample the vqe circuit and get the most probable solution
        from src.algorithms.VQE.VQE import sample_results
        highest_possible_solution = sample_results(qc, parameters, theta, backend)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "VQE Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for qaoa !!
        qc.decompose().draw("mpl", filename=vqe_circuit_image_path)

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
        most_probable_grover_result = sample_results(grover_circuit, backend)
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "Grover algorithm, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for grover !!
        grover_circuit.draw("mpl", filename=grover_circuit_image_path)
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

        #pdf.add_page()
        #pdf.set_font("Times", 'B', 16)
        #recommender_output, recommender_devices = recommender(qc)
        #pdf.cell(200, 10, "Devices recommendation based on VQE circuit", ln=True, align='L')

        # Use multi_cell to handle long text
        #pdf.set_font("Times", '', 12)  # Optionally set a smaller font for the output text
        #pdf.multi_cell(0, 10, recommender_output, align='L')

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

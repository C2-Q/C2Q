import os

import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from fpdf import FPDF
from qiskit.visualization import plot_circuit_layout

from src.algorithms.QAOA.QAOA import qaoa_no_optimization, sample_results, qaoa_optimize
from src.algorithms.VQE.VQE import vqe_optimization
from src.graph import Graph
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt
from src.problems.np_complete import NPC
from src.recommender.recommender_engine import recommender


class MaxCut(NPC):
    """
    An application class for the maximum cut problem based on a NetworkX graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """
        Args:
            graph: A graph representing the problem. It can be specified directly as a
                   NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
        """
        # supported Graph or List of edges
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

    def to_qubo(self) -> 'QUBO':
        """
        Converts the MaxCut problem into a QUBO problem represented by a QUBO class instance.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        n = len(self.nodes)
        Q = np.zeros((n, n))
        # Construct the QUBO matrix
        for i in range(n):
            node_i = self.nodes[i]
            for j in range(i + 1, n):
                node_j = self.nodes[j]
                if self.graph.has_edge(node_i, node_j):
                    # Default weight is 1
                    weight = self.graph[node_i].get(node_j).get("weight", 1)

                    # negative weights since we are minimizing
                    Q[i, i] -= weight
                    Q[j, j] -= weight
                    Q[i, j] -= -2 * weight
        return QUBO(Q)

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of node indices forming the maximum cut problem.

        Args:
            result: The calculated result of the problem (binary vector).

        Returns:
            The list of node indices whose corresponding variable is 1.
        """
        x = np.array(result)
        nodes_in_clique = []
        for idx, val in enumerate(x):
            if val == 1:
                node_label = self.indices_node[idx]
                nodes_in_clique.append(node_label)
        return nodes_in_clique

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with nodes in the maximum cut highlighted and show the weights of the edges.

        Args:
            result: The calculated result for the problem (binary vector).
            pos: The positions of nodes (optional).
        """
        x = np.array(result)
        node_colors = {}
        for idx, val in enumerate(x):
            node_label = self.indices_node[idx]
            if val == 1:
                node_colors[node_label] = 'red'
            else:
                node_colors[node_label] = 'gray'

        graph_nodes = list(self.graph.nodes())
        color_map = [node_colors[node] for node in graph_nodes]

        if pos is None:
            pos = nx.spring_layout(self.graph)

        plt.figure(figsize=(8, 6))

        nx.draw(
            self.graph,
            pos=pos,
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=12,
            font_color='white',
            edge_color='black'
        )

        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        # plt.show not included
        # plt.show()

    def report(self) -> None:
        """
        Generates a PDF report summarizing the problem, its solution, and a visualization of the result.
        """
        image_path = "graph_visualization.png"
        qaoa_circuit_image_path = "quantum_circuit_qaoa.png"
        # Create an instance of FPDF with Times New Roman font
        pdf = FPDF()
        pdf.set_font("Times", size=12)

        # New page
        pdf.add_page()

        # Set title with Times New Roman font
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "MaxCut Problem Report", ln=True, align='C')

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

        # Perform QUBO optimization and sampling using QAOA (1 layer for fast simulation)
        qubo = self.to_qubo().Q
        qaoa_dict = qaoa_optimize(qubo, layers=1)
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]
        # recommender(qc)

        # Sample the QAOA circuit and get the most probable solution
        from src.algorithms.QAOA.QAOA import sample_results
        highest_possible_solution = sample_results(qc, parameters, theta)

        # Add a new page for QAOA results
        # Draw and insert the quantum circuit (qc) into the PDF
        # for qaoa
        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization, generated quantum circuit", ln=True, align='C')
        pdf.ln(10)

        # Plot and save the quantum circuit for qaoa !!
        circuit_image_path = "quantum_circuit_qaoa.png"
        qc.draw(style="mpl")
        plt.savefig(circuit_image_path)
        plt.close()

        # Insert the quantum circuit image into the PDF
        pdf.image(circuit_image_path, x=10, y=pdf.get_y(), w=190)

        pdf.add_page()
        pdf.set_font("Times", 'B', 16)
        pdf.cell(200, 10, "QAOA Optimization Results", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Times", size=12)
        pdf.cell(200, 10, "Most Probable Solution:", ln=True, align='L')
        pdf.cell(200, 10, f"{highest_possible_solution}", ln=True, align='L')

        plt.figure(figsize=(8, 6))
        self.draw_result(highest_possible_solution, pos=pos)  # Reuse the graph positions
        qaoa_image_path_solution = "qaoa_solution_visualization.png"
        plt.savefig(qaoa_image_path_solution)
        plt.close()

        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of QAOA Solution:", ln=True, align='L')
        pdf.image(qaoa_image_path_solution, x=10, y=pdf.get_y(), w=190)

        #pdf_output_path = "maxcut_report.pdf"
        #pdf.output(pdf_output_path)

        # start here for vqe algorithm
        # Perform QUBO optimization and sampling using vqe (1 layer for fast simulation)
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

        # Plot and save the quantum circuit for vqe !!
        vqe_circuit_image_path = "vqe_quantum_circuit_qaoa.png"
        qc.draw(style="mpl")
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
        vqe_image_path_solution = "vqe_solution_visualization.png"
        plt.savefig(vqe_image_path_solution)
        plt.close()

        pdf.ln(10)
        pdf.cell(200, 10, "Visualization of VQE Solution:", ln=True, align='L')
        pdf.image(vqe_image_path_solution, x=10, y=pdf.get_y(), w=190)

        pdf_output_path = "max_cut_report.pdf"
        pdf.output(pdf_output_path)
        # clean up the saved PNG images
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(circuit_image_path):
            os.remove(circuit_image_path)

        print(f"PDF report saved as {pdf_output_path}")

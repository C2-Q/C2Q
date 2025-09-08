import os

import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from fpdf import FPDF

from src.algorithms.QAOA.QAOA import qaoa_no_optimization, qaoa_optimize, sample_results
from src.graph import Graph
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt

from src.recommender.recommender_engine import recommender
from src.reduction import clique_to_sat
from src.problems.np_complete import NPC

class Clique(NPC):
    """
    An application class for the clique problem based on a NetworkX graph.
    """

    def __init__(self, graph: nx.Graph, size: int = None) -> None:
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, Graph):
            self.graph = graph.G
        else:
            raise TypeError("The graph must be a NetworkX graph")

        self.nodes = list(self.graph.nodes())
        n = len(self.nodes)
        if n == 0:
            raise ValueError("Clique problem has an empty graph (no nodes).")

        if size is None:
            size = max(1, n - 1)
        # Clamp K safely into [1, n]
        self.size = max(1, min(size, n))

        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def to_qubo(self, A: float = 1.0, B: float = 1.0) -> 'QUBO':
        n = len(self.nodes)
        K = self.size
        if n == 0:
            raise ValueError("Cannot build QUBO for empty graph.")
        if K < 1 or K > n:
            raise ValueError(f"Invalid clique size K={K} for n={n}.")

        A = K * B + 10
        Q = np.zeros((n, n))
        linear_coeff = -2 * A * K + A
        for idx in range(n):
            Q[idx, idx] += linear_coeff
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += 2 * A
                if self.graph.has_edge(self.nodes[i], self.nodes[j]):
                    Q[i, j] += -B
        return QUBO(Q)

    def reduce_to_sat(self):
        self.sat = clique_to_sat(self.graph, self.size)


    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of node indices forming the clique.

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
        Draw the graph with nodes in the clique highlighted.
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

    def report(self) -> None:
        """
        Generates a PDF report summarizing the problem, its solution, and a visualization of the result.
        Args:
            result: The calculated result of the problem (binary vector).
        """
        # Create an instance of FPDF
        pdf = FPDF()

        # Add a page to the PDF
        pdf.add_page()

        # Set title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Clique Problem Report", ln=True, align='C')

        # Add some details about the problem
        pdf.set_font("Arial", size=12)
        pdf.ln(10)  # Add some vertical space
        pdf.cell(200, 10, f"Clique Size (K): {self.size}", ln=True, align='L')
        pdf.cell(200, 10, f"Number of Nodes: {len(self.nodes)}", ln=True, align='L')
        pdf.cell(200, 10, "Nodes in the Clique:", ln=True, align='L')

        # Interpret and list the solution using self.graph
        qubo = self.to_qubo().Q
        # qubo.display_matrix()
        # Obtain the QAOA circuit
        qaoa_dict = qaoa_optimize(qubo, layers=3)
        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]
        recommender_output, recommender_devices = recommender(qc)

        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        pdf.set_font("Arial", size=12)
        for node in self.nodes:
            pdf.cell(200, 10, f"- Node {node}", ln=True, align='L')

        # Use self.graph for visualization and save the graph with results highlighted
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph)  # Generate positions for the nodes in self.graph
        self.draw_result(highest_possible_solution, pos=pos)

        # Save the visualization as a PNG
        image_path = "clique_result.png"
        plt.savefig(image_path)
        plt.close()

        # Insert the image into the PDF
        pdf.ln(10)
        pdf.cell(200, 10, "Graph Visualization:", ln=True, align='L')
        pdf.image(image_path, x=10, y=pdf.get_y(), w=190)

        # figures = evaluate_circuit(qc)

        # Save the PDF to a file
        pdf_output_path = "clique_report.pdf"
        pdf.output(pdf_output_path)

        # Optionally, clean up the saved PNG image
        if os.path.exists(image_path):
            os.remove(image_path)

        print(f"PDF report saved as {pdf_output_path}")

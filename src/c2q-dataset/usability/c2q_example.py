import ast
import json
import random
import unittest

import networkx as nx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, MCXGate
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit_aer import AerSimulator, Aer
from qiskit_braket_provider import BraketProvider
from azure.quantum import Workspace
from qiskit_ibm_runtime import QiskitRuntimeService
from src.parser.parser import Parser, CodeVisitor, PROBLEMS



class MyTestCase(unittest.TestCase):
    def setUp(self):
        """
        NB, snippets defined withing triple quotes() can not work somehow...
        :return:
        """
        self.mul_snippet = "def a(p, q):\n    return p * q\n\n# Input json\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input json\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input json\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
        self.matrix_define = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input json\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)\nmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nnx.add_edges_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2 , 3, matrix)"
        self.sub_snippet = "def a(p, q):\n    return p - q\n\n# Input json\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.parser = Parser(model_path="../../others/saved_models")
        self.tsp_snippet = "def a(cost_matrix):\n    n = len(cost_matrix)\n    visited = [0]\n    total_cost = 0\n    current = 0\n    while len(visited) < n:\n        next_city = min([city for city in range(n) if city not in visited], key=lambda city: cost_matrix[current][city])\n        total_cost += cost_matrix[current][next_city]\n        visited.append(next_city)\n        current = next_city\n    total_cost += cost_matrix[visited[-1]][0]\n    return total_cost, visited\n\n# Input json\ncost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\ncost, route = a(cost_matrix)\nprint(cost, route)"
        self.code_visitor = CodeVisitor()
        self.clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input json\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
        self.maxCut_snippet_adj = "import networkx as nx\n\n" \
                                  "def adjacency_matrix_to_edges(matrix):\n" \
                                  "    edges = []\n" \
                                  "    for i in range(len(matrix)):\n" \
                                  "        for j in range(i + 1, len(matrix[i])):  # Use j = i + 1 to avoid duplicating edges\n" \
                                  "            if matrix[i][j] != 0:\n" \
                                  "                edges.append((i, j))  # Add edge (i, j) to the edge list\n" \
                                  "    return edges\n\n" \
                                  "def simple_cut_strategy(edges, n):\n" \
                                  "    A, B = set(), set()\n" \
                                  "    for node in range(n):\n" \
                                  "        if len(A) < len(B):\n" \
                                  "            A.add(node)\n" \
                                  "        else:\n" \
                                  "            B.add(node)\n" \
                                  "    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n" \
                                  "# Adjacency matrix as input\n" \
                                  "adjacency_matrix = [\n" \
                                  "    [0, 1, 1, 1, 1, 1],\n" \
                                  "    [1, 0, 0, 1, 0, 0],\n" \
                                  "    [1, 0, 0, 1, 0, 0],\n" \
                                  "    [1, 1, 1, 0, 1, 1],\n" \
                                  "    [1, 0, 0, 1, 0, 1],\n" \
                                  "    [1, 0, 0, 1, 1, 0]\n" \
                                  "]\n\n" \
                                  "# Convert adjacency matrix to edge list\n" \
                                  "edges = adjacency_matrix_to_edges(adjacency_matrix)\n\n" \
                                  "# Use simple_cut_strategy with the edge list and number of nodes\n" \
                                  "cut_value, A, B = simple_cut_strategy(edges, len(adjacency_matrix))\n\n" \
                                  "print(f\"Cut Value: {cut_value}\")\n" \
                                  "print(f\"Set A: {A}\")\n" \
                                  "print(f\"Set B: {B}\")\n\n" \
                                  "# Visualization of the graph\n" \
                                  "G = nx.Graph()\n" \
                                  "G.add_edges_from(edges)\n" \
                                  "pos = nx.spring_layout(G)\n" \
                                  "nx.draw(G, pos, with_labels=True, node_color=\"lightblue\", node_size=500, font_size=15)\n" \
                                  "nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): 1 for u, v in edges})  # Assuming unweighted edges\n" \
                                  "plt.show()\n"

    def test_is_pdf_generation(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        mis = PROBLEMS[problem_type](data.G)
        mis.report_latex()




if __name__ == '__main__':
    unittest.main()

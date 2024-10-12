from src.algorithms.QAOA.QAOA import qaoa_no_optimization, qaoa_optimize, sample_results
from src.parser import parser
from src.recommender.recommender_engine import recommender
from src.graph import Graph
from src.parser.parser import Parser, CodeVisitor

from src.problems.clique import Clique
from src.problems.maximal_independent_set import MIS
from src.problems.max_cut import MaxCut
from src.problems.kcolor import KColor
from src.problems.minimum_vertex_cover import MVC
from src.problems.tsp import TSP

import networkx

mul_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3),(3,4),(4,5),(5,0)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)"
matrix_define = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)\nmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nnx.add_edges_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2 , 3, matrix)"
sub_snippet = "def a(p, q):\n    return p - q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
#parser = Parser(model_path="../../others/saved_models")
tsp_snippet = "def a(cost_matrix):\n    n = len(cost_matrix)\n    visited = [0]\n    total_cost = 0\n    current = 0\n    while len(visited) < n:\n        next_city = min([city for city in range(n) if city not in visited], key=lambda city: cost_matrix[current][city])\n        total_cost += cost_matrix[current][next_city]\n        visited.append(next_city)\n        current = next_city\n    total_cost += cost_matrix[visited[-1]][0]\n    return total_cost, visited\n\n# Input data\ncost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\ncost, route = a(cost_matrix)\nprint(cost, route)"
code_visitor = CodeVisitor()
clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input data\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
maxCut_snippet_adj = "import networkx as nx\n\n" \
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

# For testing purposes, define problem type
problem = "Clique"
parser = Parser(model_path="../../others/saved_models")
if problem == "Clique":
    #problem_type, data = parser.parse(clique_snippet)
    graph = networkx.Graph()
    graph.add_nodes_from([5, 2, 3, 4, 6])
    graph.add_edges_from(([(5, 2), (4, 3), (2, 3), (5, 3), (3, 6), (4, 6)]))
    clique = Clique(graph, 3)
    qubo = clique.to_qubo().Q
elif problem == "MIS":
    problem_type, data = parser.parse(is_snippet)
    mis = MIS(data.G)
    qubo = mis.to_qubo().Q
elif problem == "MaxCut":
    problem_type, data = parser.parse(maxCut_snippet_adj)
    maxcut = MaxCut(data.G)
    qubo = maxcut.to_qubo().Q
elif problem == "KColor":
    problem_type, data = parser.parse(kcolor_snippet)
    kcolor = KColor(data.G)
    qubo = kcolor.to_qubo().Q
elif problem == "MVC":
    problem_type, data = parser.parse(mvc_snippet)
    mvc = MVC(data.G)
    qubo = mvc.to_qubo().Q
elif problem == "TSP":
    problem_type, data = parser.parse(tsp_snippet)
    tsp = TSP(data.G)
    qubo = tsp.to_qubo().Q

# Obtain the QAOA circuit
qaoa_dict = qaoa_no_optimization(qubo, layers=1)
qc = qaoa_dict["qc"]

# Run the recommender
recommender(qc)

# Run QAOA on local simulator
qaoa_dict = qaoa_optimize(qubo, layers=1)

# Obtain the parameters of the QAOA run
qc = qaoa_dict["qc"]
parameters = qaoa_dict["parameters"]
theta = qaoa_dict["theta"]

# Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
highest_possible_solution = sample_results(qc, parameters, theta)
print(f"Most probable solution: {highest_possible_solution}")

if problem == "Clique":
    clique.draw_result(highest_possible_solution)
elif problem == "MIS":
    mis.draw_result(highest_possible_solution)
elif problem == "MaxCut":
    maxcut.draw_result(highest_possible_solution)
elif problem == "KColor":
    kcolor.draw_result(highest_possible_solution)
elif problem == "MVC":
    mvc.draw_result(highest_possible_solution)
elif problem == "TSP":
    tsp.draw_result(highest_possible_solution)
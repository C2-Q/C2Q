import ast
import unittest

import networkx as nx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, MCXGate
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit_aer import AerSimulator

from src.graph import Graph
from src.algorithms.grover import grover
from src.parser.parser import Parser, CodeVisitor
from src.problems.independent_set import IS
from src.reduction import *
from src.sat_to_qubo import *
from src.circuits_library import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        """
        NB, snippets defined withing triple quotes() can not work somehow...
        :return:
        """
        self.mul_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)"
        self.matrix_define = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\nindependent_set = independent_nodes(4, edges)\nprint(independent_set)\nmatrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]\nnx.add_edges_from([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2 , 3, matrix)"
        self.sub_snippet = "def a(p, q):\n    return p - q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.parser = Parser(model_path="../../others/saved_models")
        self.tsp_snippet = "def a(cost_matrix):\n    n = len(cost_matrix)\n    visited = [0]\n    total_cost = 0\n    current = 0\n    while len(visited) < n:\n        next_city = min([city for city in range(n) if city not in visited], key=lambda city: cost_matrix[current][city])\n        total_cost += cost_matrix[current][next_city]\n        visited.append(next_city)\n        current = next_city\n    total_cost += cost_matrix[visited[-1]][0]\n    return total_cost, visited\n\n# Input data\ncost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\ncost, route = a(cost_matrix)\nprint(cost, route)"
        self.code_visitor = CodeVisitor()
        self.clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input data\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
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

    def test_something(self):
        problem_type, data = self.parser.parse(self.mul_snippet)
        self.assertEqual(problem_type, 'MUL')  # add assertion here

    def test_max_cut(self):
        problem_type, data = self.parser.parse(self.maxCut_snippet_adj)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MaxCut')  # add assertion here
        self.assertIsInstance(data.graph, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.graph), True)

    def test_tsp_snippet(self):
        problem_type, data = self.parser.parse(self.tsp_snippet)
        self.assertEqual(problem_type, 'TSP')
        data.visualize()

    def test_mis(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MIS')  # add assertion here
        self.assertIsInstance(data.graph, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.graph), True)
        ism = IS(data, 2)
        print(ism.sat.clauses)
        print(solve_all_cnf_solutions(ism.sat))
        ism.reduce_to_3sat()
        counts = ism.grover()
        print(counts)
        plot_histogram(counts)
        plt.show()

    def test_mul(self):
        problem_type, data = self.parser.parse(self.mul_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MUL')

    def test_sub(self):
        problem_type, data = self.parser.parse(self.sub_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'SUB')

    def test_codeVisitor(self):
        tree = ast.parse(self.mul_snippet)
        self.code_visitor.visit(tree)
        print(self.code_visitor.get_extracted_data())
        print(self.code_visitor.function_calls)

    def test_clique_snippet(self):
        problem_type, data = self.parser.parse(self.clique_snippet)
        print(problem_type, data)
        cnf = clique_to_sat(data.graph, 3)
        data.visualize()
        sat = sat_to_3sat(cnf)
        print(f'clauses before conversion: {len(cnf.clauses)}')
        print(f'clauses after conversion: {len(sat.clauses)}')
        print(cnf.clauses)
        print(sat.clauses)
        print(solve_all_cnf_solutions(cnf))
        cha = Chancellor(sat)
        cha.fillQ()
        #cha.solveQ()

    def test_clique(self):
        self.assertEqual(True, True)

    def test_graph_init(self):
        # Example 1: Using a distance matrix
        distance_matrix = [
            [0, 2, 3, 0],
            [2, 0, 4, 6],
            [3, 4, 0, 5],
            [0, 6, 5, 0]
        ]
        graph_matrix = Graph(input_data=distance_matrix)
        graph_matrix.visualize()

        # Example 2: Using a list of edges with weights
        edges_with_weights = [(0, 1, 2), (0, 2, 3), (1, 2, 4), (1, 3, 6), (2, 3, 5)]
        graph_edges_with_weights = Graph(input_data=edges_with_weights)
        graph_edges_with_weights.visualize()

        # Example 3: Using a list of edges without weights (default weight of 1 will be assigned)
        edges_without_weights = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        graph_edges_without_weights = Graph(input_data=edges_without_weights)
        graph_edges_without_weights.visualize()

        # Example 4: Generate a random graph
        random_graph = Graph.random_graph(num_nodes=5, edge_prob=0.5, weighted=True)
        random_graph.visualize()

        # Example 5: Invalid input
        invalid_input = {"a": 1, "b": 2}  # Not a valid format
        try:
            graph_invalid = Graph(input_data=invalid_input)
        except ValueError as e:
            print(f"Error: {e}")

        self.assertEqual(True, True)

    def test_2_3sat(self):
        # Example usage:
        cnf = [[-1, -2, -3], [1, 2, 3], [1, -2, -3], [1, 3, -2], [-1, -2, 3], [2, 1, 4],
               [-1, -3, -4], [2, 3, 4], [1, 3, 4], [1, -3, 4], [1, 3, 5], [3, 4, 5], [-2, -3, -5], [1, 2, -3],
               [2, 4, 5], [2, 3, 5], [-1, -4, -5], [3, 4]]
        #cnf = [[1, 2, 3], [2, -3], [-2], [1, 2, 3, 4], [3, 4, 5], [2, 3, 4], [2, 3, 5], [3, 4, 6], [3, 4, -5, -6]]
        converted_cnf = sat_to_3sat(cnf)
        print("Converted 3-SAT CNF:", converted_cnf.clauses)

        # Find all solutions for the original and converted CNF
        original_solutions = solve_all_cnf_solutions(cnf)
        print(len(original_solutions))
        converted_solutions = solve_all_cnf_solutions(converted_cnf)
        print(len(converted_solutions))

        cha = Chancellor(converted_cnf.clauses)
        cha.solveQ()
        print("Original CNF solutions:", original_solutions)
        print("Converted CNF solutions:", converted_solutions)
        cha.printQ()

    def test_chancellor(self):
        # Define CNF formula using clauses
        clauses = [
            [1, 2, 3, 4],
            [1, -2, 3],
            [1, 2],
            [-2],
            [-3],
            [-2, 3, 4]
        ]
        # Create a CNF object from the clauses
        formula = CNF(from_clauses=clauses)
        print("solutions:")
        print(solve_all_cnf_solutions(formula))
        chancellor_instance = Chancellor(formula)

        # Fill the QUBO matrix based on the formula
        chancellor_instance.fillQ()

        # Print the resulting QUBO matrix
        print("QUBO Matrix:")
        for key, value in chancellor_instance.Q.items():
            print(f"Q[{key}] = {value}")

        # Visualize the QUBO matrix
        chancellor_instance.visualizeQ()

        # Solve the QUBO problem
        chancellor_instance.solveQ()

    def test_is_chancellor(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        # Convert to SAT problem with an independent set of size 2
        independent_set_cnf = independent_set_to_sat(G, 3)
        # Print the CNF clauses
        print("CNF Clauses for Independent Set Problem:")
        for clause in independent_set_cnf.clauses:
            print(clause)

        print("Solution:")
        print(solve_all_cnf_solutions(independent_set_cnf))
        converted_cnf = sat_to_3sat(independent_set_cnf)
        print(len(independent_set_cnf.clauses))
        print(len(converted_cnf.clauses))
        print(converted_cnf.clauses)
        ch = Chancellor(converted_cnf)
        ch.fillQ()
        ch.visualizeQ()
        ch.solveQ()

    def test_grover_operator(self):
        # Example usage:

        # Define the oracle for Grover's algorithm
        # This oracle flips the phase of the state |11> (marked state)
        num_qubits = 2
        oracle = QuantumCircuit(num_qubits)
        oracle.cz(0, 1)  # Apply a controlled-Z gate to the state |11>
        oracle.name = "Oracle"

        grover_circuit = grover(oracle, iterations=1)
        backend = AerSimulator()

        grover_circuit.draw('mpl')
        transpiled_circuit = transpile(grover_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()

    def test_cnf_circuits(self):
        # Define CNF formula using clauses
        clauses = [
            [-1, -2], [-2, -3], [-3, -4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]
        ]
        # Create a CNF object from the clauses
        formula = CNF(from_clauses=clauses)
        self.assertEqual(True, True)
        qc = cnf_to_quantum_circuit_optimized(formula)
        qc.measure_all()
        combined_circuit = QuantumCircuit(qc.num_qubits)
        combined_circuit.h([0, 1, 2, 3])
        combined_circuit = combined_circuit.compose(qc)
        backend = AerSimulator()
        transpiled_circuit = transpile(combined_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()
        print(counts)

    def test_3sat_oracle(self):
        # Sample CNF clauses for testing
        clauses = [
            [-1, -2], [-2, -3], [-3, -4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]
        ]

        # Convert clauses to CNF formula
        formula = CNF(from_clauses=clauses)
        qc = cnf_to_quantum_oracle_optimized(formula)

        combined_circuit = QuantumCircuit(qc.num_qubits)
        #combined_circuit.h(range(formula.nv))
        combined_circuit.x([1, 3])
        combined_circuit = combined_circuit.compose(qc)

        state = Statevector(combined_circuit)

        # Print out the amplitudes and probabilities for each state
        for idx, amplitude in enumerate(state):
            binary_state = format(idx, f'0{qc.num_qubits}b')  # Convert index to binary representation
            print(f"State {binary_state}: Amplitude = {amplitude}, Probability = {abs(amplitude) ** 2}")

    def test_is_oracle(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        # Convert to SAT problem with an independent set of size 2
        independent_set_cnf = independent_set_to_sat(data.graph, 2)
        oracle = cnf_to_quantum_oracle_optimized(independent_set_cnf)
        combined_circuit = QuantumCircuit(oracle.num_qubits)
        # combined_circuit.h(range(formula.nv))
        combined_circuit.x([0, 2])
        data.visualize()
        combined_circuit = combined_circuit.compose(oracle)

        state = Statevector(combined_circuit)

        # Print out the amplitudes and probabilities for each state
        for idx, amplitude in enumerate(state):
            binary_state = format(idx, f'0{oracle.num_qubits}b')  # Convert index to binary representation
            print(f"State {binary_state}: Amplitude = {amplitude}, Probability = {abs(amplitude) ** 2}")

    def test_is_grover(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3)])
        # Convert to SAT problem with an independent set of size 2
        independent_set_cnf = independent_set_to_sat(data.graph, 2)
        oracle = cnf_to_quantum_oracle_optimized(independent_set_cnf)
        grover_circuit = grover(oracle, objective_qubits=[0, 1, 2, 3], iterations=1)
        from qiskit.circuit.library import GroverOperator
        op = GroverOperator(oracle,
                            reflection_qubits=[0, 1, 2, 3])
        qr = QuantumRegister(op.num_qubits)
        cr = ClassicalRegister(4)
        circuit = QuantumCircuit(qr, cr)
        circuit.h([0, 1, 2, 3])

        circuit = circuit.compose(op)
        circuit.measure([0, 1, 2, 3], cr)
        print(solve_all_cnf_solutions(independent_set_cnf))
        #print(op.decompose())
        print(grover_circuit)
        backend = AerSimulator()
        transpiled_circuit = transpile(grover_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()
        print(counts)

    def test_cnf_grover(self):
        cnf = [[-1, -2, -3], [1, 2, 3], [1, -2, -3], [1, 3, -2], [-1, -2, 3], [2, 1, 4],
               [-1, -3, -4], [2, 3, 4], [1, 3, 4], [1, -3, 4], [1, 3, 5], [3, 4, 5], [-2, -3, -5], [1, 2, -3],
               [2, 4, 5], [2, 3, 5], [-1, -4, -5], [3, 4]]

        # Convert clauses to CNF formula
        formula = CNF(from_clauses=cnf)
        print(solve_all_cnf_solutions(formula))
        oracle = cnf_to_quantum_oracle_optimized(formula)
        grover_circuit = grover(oracle, iterations=3, objective_qubits=[0, 1, 2, 3, 4])
        backend = AerSimulator()
        transpiled_circuit = transpile(grover_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()


if __name__ == '__main__':
    unittest.main()

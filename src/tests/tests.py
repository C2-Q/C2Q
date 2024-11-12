import ast
import random
import unittest

import networkx as nx
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, MCXGate
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit_aer import AerSimulator, Aer

from src.algorithms.QAOA.QAOA import convert_qubo_to_ising, qaoa_optimize, qaoa_no_optimization, sample_results
from src.algorithms.VQE.VQE import vqe_optimization
from src.graph import Graph
from src.algorithms.grover import grover
from src.parser.parser import Parser, CodeVisitor
from src.problems.Three_SAT import ThreeSat
from src.problems.clique import Clique
from src.problems.factorization import Factor
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.tsp import TSP
from src.recommender.recommender_engine import recommender, plot_results
from src.reduction import *
from src.sat_to_qubo import *
from src.circuits_library import *


def validate_assignment(formula, assignment):
    """Check if the given assignment satisfies the CNF formula."""
    # Adjust to access assignment values as a tuple or list
    # assignment[i - 1] gives the assignment for variable i
    for clause in formula.clauses:
        clause_satisfied = False
        for literal in clause:
            var = abs(literal)
            value = assignment[var - 1]  # Get the assigned value from the tuple
            if (literal > 0 and value == 1) or (literal < 0 and value == 0):
                clause_satisfied = True
                break
        if not clause_satisfied:
            return False
    return True


def generate_random_cnf(num_vars, num_clauses, max_literals_per_clause):
    """Generate a random CNF formula with the specified parameters."""
    clauses = []

    for _ in range(num_clauses):
        clause = set()

        while len(clause) < max_literals_per_clause:
            literal = random.randint(1, num_vars)  # Choose a random variable
            if random.choice([True, False]):
                literal = -literal  # Randomly negate the literal

            # Check for contradictory or duplicate literals in the clause
            if literal not in clause and -literal not in clause:
                clause.add(literal)

        clauses.append(list(clause))

    return CNF(from_clauses=clauses)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        """
        NB, snippets defined withing triple quotes() can not work somehow...
        :return:
        """
        self.mul_snippet = "def a(p, q):\n    return p * q\n\n# Input data\np, q = -8, 8\nresult = a(-10, q)\nprint(result)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
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

    def test_pdf_generation(self):
        problem_type, data = self.parser.parse(self.maxCut_snippet_adj)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MaxCut')  # add assertion here
        self.assertIsInstance(data.G, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.G), True)
        mc = MaxCut(data.G)
        mc.report()

    def test_is_pdf_generation(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MIS')  # add assertion here
        self.assertIsInstance(data.G, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.G), True)
        mis = MIS(data.G)
        mis.report()

        formula = maximal_independent_set_to_sat(data.G)
        formula = sat_to_3sat(formula)
        sat = ThreeSat(formula)
        qubo = sat.to_qubo()
        qubo.display_matrix()
        sat.report()


    def test_max_cut(self):
        problem_type, data = self.parser.parse(self.maxCut_snippet_adj)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MaxCut')  # add assertion here
        self.assertIsInstance(data.G, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.G), True)
        clique = Clique(data.G, 4)
        qubo = clique.to_qubo()
        qubo.display_matrix()

        # Obtain the QAOA circuit
        qubo = qubo.Q
        qaoa_dict = qaoa_no_optimization(qubo, layers=1)
        qc = qaoa_dict["qc"]

        # Run the recommender
        recommender_output, recommender_devices = recommender(qc)
        print(recommender_output)

        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        clique.draw_result(highest_possible_solution)

    def test_max_cut_vqe(self):
        problem_type, data = self.parser.parse(self.maxCut_snippet_adj)
        print(problem_type, data)
        self.assertEqual(problem_type, 'MaxCut')  # add assertion here
        self.assertIsInstance(data.G, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.G), True)
        clique = Clique(data.G, 3)
        qubo = clique.to_qubo()
        qubo.display_matrix()
        print(qubo.solve_brute_force())
        # Obtain the QAOA circuit
        qubo = qubo.Q

        # qaoa_dict = qaoa_no_optimization(qubo, layers=1)
        # qc = qaoa_dict["qc"]
        # # Run the recommender
        # recommender(qc)

        # Run QAOA on local simulator
        vqe_dict = qaoa_optimize(qubo, layers=5)

        # Obtain the parameters of the QAOA run
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]
        recommender(qc)
        from src.algorithms.VQE.VQE import sample_results
        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        clique.draw_result(highest_possible_solution)

    def test_tsp_snippet(self):
        problem_type, data = self.parser.parse(self.tsp_snippet)
        tsp = TSP(data.G)
        self.assertEqual(problem_type, 'TSP')
        data.visualize()
        qubo = tsp.to_qubo()

        # Obtain the QAOA circuit
        qubo = qubo.Q
        qaoa_dict = qaoa_no_optimization(qubo, layers=1)
        qc = qaoa_dict["qc"]

        # Run the recommender
        recommender_output, recommender_devices = recommender(qc)
        print(recommender_output)

        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        tsp.draw_result(highest_possible_solution)

    def test_mis(self):
        problem_type, data = self.parser.parse(self.is_snippet)
        print(problem_type, data)
        ims = MIS(data.G)
        print(data.G.nodes)
        self.assertEqual(problem_type, 'MIS')  # add assertion here
        self.assertIsInstance(data.G, nx.Graph)
        data.visualize()
        self.assertEqual(nx.is_weighted(data.G), True)
        qubo = ims.to_qubo()
        qubo.display_matrix()

        # Obtain the QAOA circuit
        qubo = qubo.Q
        qaoa_dict = qaoa_no_optimization(qubo, layers=4)
        qc = qaoa_dict["qc"]

        # Run the recommender
        recommender_output, recommender_devices = recommender(qc)
        print(recommender_output)

        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        ims.draw_result(highest_possible_solution)

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
        clique = Clique(data.G, 3)
        cnf = clique_to_sat(data.G, 3)
        data.visualize()
        sat = sat_to_3sat(cnf)
        print(solve_all_cnf_solutions(cnf))
        cha = Chancellor(sat)
        cha.fillQ()
        qubo = clique.to_qubo()
        qubo.display_matrix()
        op, offset = convert_qubo_to_ising(qubo.Q)
        print(op, offset)

        # Obtain the QAOA circuit
        qubo = qubo.Q
        qaoa_dict = qaoa_no_optimization(qubo, layers=1)
        qc = qaoa_dict["qc"]

        # Run the recommender
        recommender_output, recommender_devices = recommender(qc)
        print(recommender_output)

        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=3)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        clique.draw_result(highest_possible_solution)

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
        # cnf = [[1, 2, 3], [2, -3], [-2], [1, 2, 3, 4], [3, 4, 5], [2, 3, 4], [2, 3, 5], [3, 4, 6], [3, 4, -5, -6]]
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

    def test_cnf_circuit(self):
        clauses = [
            [1, 2, 3],
            [1, -2, 3],
            [1, -3],
            [-1, 4]
        ]
        formula = CNF(from_clauses=clauses)
        print(solve_all_cnf_solutions(formula))
        sat = ThreeSat(formula)
        qubo = sat.to_qubo()
        qubo.display_matrix()
        sat.report()

        # qaoa_dict = qaoa_optimize(qubo.Q, layers=2)
        # qaoa_qc = qaoa_dict["qc"]
        # qaoa_parameters = qaoa_dict["parameters"]
        # qaoa_theta = qaoa_dict["theta"]
        # from src.algorithms.QAOA.QAOA import sample_results
        # qaoa_solution = sample_results(qaoa_qc, qaoa_parameters, qaoa_theta)
        # print(qaoa_solution)
        # print("brute force")
        # print(qubo.solve_brute_force())
        #
        # # VQE Solution
        # vqe_dict = vqe_optimization(qubo.Q, layers=3)
        # vqe_qc = vqe_dict["qc"]
        # vqe_parameters = vqe_dict["parameters"]
        # vqe_theta = vqe_dict["theta"]
        # from src.algorithms.VQE.VQE import sample_results
        # vqe_solution = sample_results(vqe_qc, vqe_parameters, vqe_theta)
        # print(vqe_solution)
        # grover_circuit = sat.grover_sat(iterations=3)
        # from src.algorithms.grover import sample_results
        # grover_solution = sample_results(grover_circuit)
        # print(grover_solution)
    def test_chancellor_randomized(self):
        num_correct = 0
        num_tests = 50
        num_vars = 7
        num_clauses = 10
        max_literals_per_clause = 3
        for test_num in range(num_tests):
            print(f"\nRunning test case {test_num + 1}")

            # Generate a random CNF formula
            formula = generate_random_cnf(num_vars, num_clauses, max_literals_per_clause)
            print("Generated CNF clauses:", formula.clauses)
            print(solve_all_cnf_solutions(formula))
            # Initialize the Chancellor instance
            formula = sat_to_3sat(formula)
            chancellor_instance = Chancellor(formula)

            # Fill the QUBO matrix based on the formula
            chancellor_instance.fillQ()

            # Print the resulting QUBO matrix
            print("QUBO Matrix:")
            for key, value in chancellor_instance.Q.items():
                print(f"Q[{key}] = {value}")

            # Solve the QUBO problem
            best_assignment, _ = chancellor_instance.solveQ()

            # Validate best_assignment
            is_correct = validate_assignment(formula, best_assignment)
            # print("Best assignment:", best_assignment)
            if is_correct: num_correct += 1
            print("Is the assignment correct?", "Yes" if is_correct else "No")
        print(num_correct)

    def test_chancellor(self):
        # Define CNF formula using clauses
        clauses = [
            [1, 2, 3, 4],
            [1, -2, 3],
            [1, 2],
            [4, 5],
            [2],
            [-2, 3],
            [-2, 3, 4],
            [3,4,6],
            [6,-3, -4]
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
        best_assignment, _ = chancellor_instance.solveQ()

    def test_is_chancellor(self):
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        # Convert to SAT problem with an independent set of size 2
        independent_set_cnf = maximal_independent_set_to_sat(G)
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
        # combined_circuit.h(range(formula.nv))
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
        independent_set_cnf = independent_set_to_k_sat(data.G, 2)
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
        G.add_edges_from([(0, 1), (0, 2), (1, 2)])
        graph = Graph([(0, 1), (0, 2), (1, 2)])
        graph.visualize()
        # Convert to SAT problem with an independent set of size 2
        independent_set_cnf = independent_set_to_k_sat(data.G, 1)
        oracle = cnf_to_quantum_oracle_optimized(independent_set_cnf)
        state_prep = QuantumCircuit(oracle.num_qubits)
        state_prep.h([0, 1, 2])
        grover_circuit = grover(oracle, objective_qubits=[0, 1, 2],
                                working_qubits=[0, 1, 2], state_pre=state_prep
                                , iterations=1)
        oracle.draw('mpl')
        plt.show()
        print(solve_all_cnf_solutions(independent_set_cnf))
        # print(op.decompose())
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
        state_prep = QuantumCircuit(oracle.num_qubits)
        state_prep.h([0, 1, 2, 3, 4])
        grover_circuit = grover(oracle, iterations=3,
                                objective_qubits=[0, 1, 2, 3, 4],
                                state_pre=state_prep,
                                working_qubits=[0, 1, 2, 3, 4]
                                )
        backend = AerSimulator()
        transpiled_circuit = transpile(grover_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        print(counts)
        plot_histogram(counts)
        plt.show()

    def test_factor(self):
        factor = Factor(35)
        qc = factor.grover(iterations=2)
        print(qc)
        backend = AerSimulator()
        transpiled_circuit = transpile(qc, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()

    def test_plot_recommender(self):
        # Plot recommender results with a range of qubits

        recommender_data_array = []
        qubits_array = []

        # Define the range of qubits
        for z in range(4, 21, 2):
            qubits_array.append(z)

            # Generate 3-regular graphs.
            G = nx.random_regular_graph(3, z, seed=100)

            # Turn 3-regular graphs into MaxCut QUBO formulation (can be any other problem too)
            maxcut = MaxCut(G)
            qubo = maxcut.to_qubo()
            qubo = qubo.Q

            qaoa_dict = qaoa_no_optimization(qubo, layers=1)
            qc = qaoa_dict["qc"]

            # Run the recommender and append recommender_data_array
            recommender_output, recommender_devices = recommender(qc)
            recommender_data_array.append(recommender_devices)

        plot_results(recommender_data_array, qubits_array)


if __name__ == '__main__':
    unittest.main()

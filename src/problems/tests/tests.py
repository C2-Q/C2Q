import unittest

import networkx
import numpy as np
from matplotlib import pyplot as plt
from qiskit import transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

from src.algorithms.VQE.VQE import vqe_optimization
from src.graph import Graph
from src.problems.basic_arithmetic.multiplication import Mul
from src.problems.clique import Clique
from src.problems.factorization import Factor
from src.problems.kcolor import KColor
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.minimum_vertex_cover import MVC
from src.problems.qubo import QUBO
from src.problems.tsp import TSP


class MyTestCase(unittest.TestCase):
    def test_something(self):
        Q = np.array([
            [2, -1, 0],
            [0, 2, -1],
            [0, 0, 2]
        ])
        qubo = QUBO(Q)
        qubo.display()
        from itertools import product
        for x in product([0, 1], repeat=qubo.n):
            x_array = np.array(x)
            value = qubo.evaluate(x_array)
            print(f"{x_array} : {value}")
        self.assertEqual(True, True)

    def test_cliques(self):
        graph = networkx.Graph()
        graph.add_nodes_from([5, 2, 3, 4, 6])
        graph.add_edges_from(([(5, 2), (4, 3), (2, 3), (5, 3), (3, 6), (4, 6)]))
        #graph = Graph.random_graph(num_nodes=5)
        clique_problem = Clique(graph, size=3)
        #graph.visualize()
        qubo_instance = clique_problem.to_qubo()
        qubo_instance.display()
        qubo_instance.display_matrix()
        result = qubo_instance.solve_brute_force()
        list = clique_problem.interpret(result[0])
        print(result[0])
        print(list)
        clique_problem.draw_result(result[0])

    def test_ims(self):
        graph = Graph.random_graph(num_nodes=6)
        print(graph.nodes)
        weight = graph[1]  # Default weight is 1
        print(weight)
        mis = MIS(graph)
        # graph.visualize()
        qubo_instance = mis.to_qubo()
        qubo_instance.display()
        qubo_instance.display_matrix()
        result = qubo_instance.solve_brute_force()
        list = mis.interpret(result[0])
        mis.draw_result(result[0])

    def test_max_cut(self):
        graph = Graph.random_graph(num_nodes=10)
        self.assertEqual(True, True)
        maxcut = MaxCut(graph)
        qubo = maxcut.to_qubo()
        qubo.display_matrix()
        result = qubo.solve_brute_force()
        list = maxcut.interpret(result[0])
        maxcut.draw_result(result[0])

    def test_tsp(self):
        graph = Graph.random_graph(num_nodes=4)
        print(graph.adj)
        self.assertEqual(True, True)
        tsp = TSP(graph)
        qubo = tsp.to_qubo()
        qubo.display_matrix()
        result = qubo.solve_brute_force()
        print(result[0])
        list = tsp.interpret(result[0])
        print(list)
        tsp.draw_result(result[0])

    def test_vc(self):
        graph = Graph.random_graph(num_nodes=6)
        vc = MVC(graph)
        qubo_instance = vc.to_qubo()
        qubo_instance.display()
        qubo_instance.display_matrix()
        result = qubo_instance.solve_brute_force()
        qubo = qubo_instance.Q
        # Run QAOA on local simulator
        vqe_dict = vqe_optimization(qubo, layers=2)

        # Obtain the parameters of the QAOA run
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]
        from src.algorithms.VQE.VQE import sample_results
        # Sample the QAOA circuit with optimized parameters and obtain the most probable solution based on the QAOA run
        highest_possible_solution = sample_results(qc, parameters, theta)
        print(f"Most probable solution: {highest_possible_solution}")
        vc.draw_result(highest_possible_solution)
        print(result[0])
        list = vc.interpret(result[0])
        print(list)

    def test_k_coloring(self):
        print("Most probable solution: [1 0 1 1 0]")
        graph = networkx.Graph()
        graph.add_nodes_from([5, 2, 3, 4, 6])
        graph.add_edges_from(([(5, 2), (4, 3), (2, 3), (5, 3), (3, 6), (4, 6)]))
        graph = Graph.random_graph(num_nodes=6)
        self.assertEqual(True, True)
        kc = KColor(graph, 3)
        qubo = kc.to_qubo()
        qubo.display_matrix()
        result = qubo.solve_brute_force()
        list = kc.interpret(result[0])
        print(result[0])
        print(list)
        kc.draw_result(result[0])

    def test_mul(self):
        mul = Mul(1, 1)
        self.assertEqual(True, True)
        qc = mul.quantum_circuit()
        latex_code = qc.decompose().draw(output="latex_source")
        print(latex_code)
        # sampler = Sampler()
        # result = sampler.run(qc).result()
        # result_counts = result.quasi_dists[0]
        # result_value = max(result_counts, key=result_counts.get)
        # print(result_value)

    def test_factorization(self):
        factor = Factor(24)
        qc = factor.grover(iterations=2)
        print(qc)
        backend = AerSimulator()
        transpiled_circuit = transpile(qc, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        plot_histogram(counts)
        plt.show()

    def test_factor_grover(self):
        factor = Factor(18)
        print(factor.execute())


if __name__ == '__main__':
    unittest.main()

import unittest

import networkx
import numpy as np

from src.graph import Graph
from src.problems.clique import Clique
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
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
        graph = Graph.random_graph(num_nodes=5)
        clique_problem = Clique(graph, size=4)
        #graph.visualize()
        qubo_instance = clique_problem.to_qubo()
        qubo_instance.display()
        qubo_instance.display_matrix()
        result = qubo_instance.solve_brute_force()
        list = clique_problem.interpret(result[0])
        clique_problem.draw_result(result[0])

    def test_ims(self):
        graph = Graph.random_graph(num_nodes=6)
        print(graph.nodes)
        weight = graph[1] # Default weight is 1
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
        graph = Graph.random_graph(num_nodes=3)
        print(graph.adj)
        self.assertEqual(True, True)
        tsp = TSP(graph)
        qubo = tsp.to_qubo()
        qubo.display_matrix()
        # result = qubo.solve_brute_force()
        # print(result)
        # list = tsp.interpret(result[0])
        # tsp.draw_result(result[0])


if __name__ == '__main__':
    unittest.main()

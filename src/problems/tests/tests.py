import unittest

import networkx
import numpy as np

from src.graph import Graph
from src.problems.clique import Clique
from src.problems.qubo import QUBO


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
        clique_problem = Clique(graph, size=3)
        #graph.visualize()
        qubo_instance = clique_problem.to_qubo()
        qubo_instance.display()
        qubo_instance.display_matrix()
        result = qubo_instance.solve_brute_force()
        list = clique_problem.interpret(result[0])
        clique_problem.draw_result(result[0])


if __name__ == '__main__':
    unittest.main()

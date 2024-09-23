import unittest

import numpy as np

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


if __name__ == '__main__':
    unittest.main()

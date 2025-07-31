import copy
import unittest
from pysat.formula import CNF
from src.reduction import sat_to_3sat, solve_all_cnf_solutions


class SatTo3SatTests(unittest.TestCase):
    def test_does_not_mutate_input(self):
        cnf = CNF(from_clauses=[[1, 2, 3], [1, -2]])
        original = copy.deepcopy(cnf.clauses)
        _ = sat_to_3sat(cnf)
        self.assertEqual(cnf.clauses, original)

    def test_padding_of_short_clauses(self):
        cnf = CNF(from_clauses=[[1], [-2, 3]])
        converted = sat_to_3sat(cnf)
        self.assertEqual(converted.clauses, [[1, 1, 1], [-2, 3, 3]])

    def test_complex_formula_equivalence(self):
        cnf = CNF(from_clauses=[
            [1, -2, 3, 4, 5],
            [-1, 2, -3],
            [4],
            [5, -4],
            [-2, -3, -5, 1]
        ])
        expected = solve_all_cnf_solutions(cnf)
        converted = sat_to_3sat(cnf)
        actual = solve_all_cnf_solutions(converted)
        num_vars = max(abs(l) for clause in cnf.clauses for l in clause)
        proj = lambda sol: [lit for lit in sol if abs(lit) <= num_vars]

        self.assertEqual(sorted(map(proj, expected)), sorted(map(proj, actual)))
        for clause in converted.clauses:
            self.assertEqual(len(clause),3)


if __name__ == "__main__":
    unittest.main()

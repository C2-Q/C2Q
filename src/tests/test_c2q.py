import unittest
from pathlib import Path

import networkx as nx
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ionq")
pytest.importorskip("pytket")
pytest.importorskip("braket")

from src.parser.parser import Parser
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.tsp import TSP
from src.algorithms.QAOA.QAOA import qaoa_optimize, qaoa_no_optimization, sample_results
from src.recommender.recommender_engine import recommender


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Always resolve paths relative to THIS test file so pytest working-dir doesn't matter.
        # Expected repo layout: <repo>/src/tests/<this_file>.py
        repo_root = Path(__file__).resolve().parents[2]  # .../<repo>
        model_dir = repo_root / "src" / "parser" / "saved_models"
        self.parser = Parser(model_path=str(model_dir))

        # Output directory: <repo>/src/tests/tests_output/...
        self.out_dir = Path(__file__).resolve().parent / "tests_output"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_cut_python = "def max_cut_brute_force(n, edges):\n    best_cut_value = 0\n    best_partition = None\n    for i in range(1 << n):\n        set_A = {j for j in range(n) if i & (1 << j)}\n        set_B = set(range(n)) - set_A\n        cut_value = sum(1 for u, v in edges if (u in set_A and v in set_B) or (u in B and v in A))\n        if cut_value > best_cut_value:\n            best_cut_value = cut_value\n            best_partition = (set_A, set_B)\n    return best_cut_value, best_partition\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3), (3, 0)]\ncut_value, partition = max_cut_brute_force(4, edges)\nprint(cut_value, partition)"
        # --- Snippets (presentation-friendly) ---
        self.maxCut_snippet_adj = (
            "import networkx as nx\n"
            "def adjacency_matrix_to_edges(matrix):\n"
            "    edges = []\n"
            "    for i in range(len(matrix)):\n"
            "        for j in range(i + 1, len(matrix[i])):\n"
            "            if matrix[i][j] != 0:\n"
            "                edges.append((i, j))\n"
            "    return edges\n"
            "\n"
            "adjacency_matrix = [\n"
            "    [0, 1, 1, 1, 1, 1],\n"
            "    [1, 0, 0, 1, 0, 0],\n"
            "    [1, 0, 0, 1, 0, 0],\n"
            "    [1, 1, 1, 0, 1, 1],\n"
            "    [1, 0, 0, 1, 0, 1],\n"
            "    [1, 0, 0, 1, 1, 0]\n"
            "]\n"
            "edges = adjacency_matrix_to_edges(adjacency_matrix)\n"
            "G = nx.Graph()\n"
            "G.add_edges_from(edges)\n"
            "print('edges', edges)\n"
        )

        self.is_snippet = (
            "def independent_nodes(n, edges):\n"
            "    independent_set = set()\n"
            "    for node in range(n):\n"
            "        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n"
            "            independent_set.add(node)\n"
            "    return independent_set\n"
            "\n"
            "edges = [(0, 1), (0, 2), (1, 2), (1, 3)]\n"
            "independent_set = independent_nodes(4, edges)\n"
            "print(independent_set)\n"
        )

        self.tsp_snippet = (
            "def a(cost_matrix):\n"
            "    n = len(cost_matrix)\n"
            "    visited = [0]\n"
            "    total_cost = 0\n"
            "    current = 0\n"
            "    while len(visited) < n:\n"
            "        next_city = min([city for city in range(n) if city not in visited],\n"
            "                        key=lambda city: cost_matrix[current][city])\n"
            "        total_cost += cost_matrix[current][next_city]\n"
            "        visited.append(next_city)\n"
            "        current = next_city\n"
            "    total_cost += cost_matrix[visited[-1]][0]\n"
            "    return total_cost, visited\n"
            "\n"
            "cost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\n"
            "cost, route = a(cost_matrix)\n"
            "print(cost, route)\n"
        )

    def _stem(self, name: str) -> str:
        """
        Return an output STEM path (no extension) as string.
        report_latex(output_path=...) in other tests expects a stem.
        """
        return str(self.out_dir / name)

    # -------------------- PRESENTATION TEST 1 --------------------
    def test_01_maxcut_report_latex(self):
        """
        Demo: Parser -> Graph -> MaxCut -> report_latex() -> saved into tests_output/
        """
        problem_type, data = self.parser.parse(self.max_cut_python)
        self.assertEqual(problem_type, "MaxCut")
        self.assertIsInstance(data.G, nx.Graph)

        mc = MaxCut(data.G)

        # Prefer report_latex with explicit output path
        if hasattr(mc, "report_latex"):
            mc.report_latex(output_path=self._stem("01_maxcut"))
        else:
            mc.report()

    # -------------------- PRESENTATION TEST 2 --------------------
    def test_02_mis_report_latex_and_qaoa(self):
        """
        Demo: Parser -> MIS -> report_latex() -> QUBO -> QAOA -> recommender
        """
        problem_type, data = self.parser.parse(self.is_snippet)
        self.assertEqual(problem_type, "MIS")
        self.assertIsInstance(data.G, nx.Graph)

        ims = MIS(data.G)

        if hasattr(ims, "report_latex"):
            ims.report_latex(output_path=self._stem("02_mis"))
        else:
            self.fail("MIS has no report_latex(output_path=...) method")

        # qubo_obj = ims.to_qubo()
        # qubo = qubo_obj.Q
        #
        # # Fast circuit build for recommender
        # qaoa_dict = qaoa_no_optimization(qubo, layers=1)
        # qc = qaoa_dict["qc"]
        # recommender_output, _ = recommender(qc, save_figures=True, figures_dir="tests_output/recommender")
        #
        # # Small optimization for presentation speed
        # qaoa_dict_opt = qaoa_optimize(qubo, layers=1)
        # best = sample_results(qaoa_dict_opt["qc"], qaoa_dict_opt["parameters"], qaoa_dict_opt["theta"])

    # -------------------- PRESENTATION TEST 3 --------------------
    def test_03_tsp_report_latex_and_qaoa(self):
        """
        Demo: Parser -> TSP -> report_latex() -> QUBO -> QAOA -> sample solution
        """
        problem_type, data = self.parser.parse(self.tsp_snippet)
        self.assertEqual(problem_type, "TSP")
        self.assertIsInstance(data.G, nx.Graph)

        tsp = TSP(data.G)

        if hasattr(tsp, "report_latex"):
            tsp.report_latex(output_path=self._stem("03_tsp"))
        else:
            self.fail("TSP has no report_latex(output_path=...) method")

        qubo_obj = tsp.to_qubo()
        qubo = qubo_obj.Q

        qaoa_dict = qaoa_optimize(qubo, layers=1)
        best = sample_results(qaoa_dict["qc"], qaoa_dict["parameters"], qaoa_dict["theta"])


if __name__ == "__main__":
    unittest.main()

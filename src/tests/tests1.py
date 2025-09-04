import os
import unittest
import ast
import json
import random
import unittest
from collections import defaultdict

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
from src.parser.parser import Parser, CodeVisitor, PROBLEMS
from src.problems.Three_SAT import ThreeSat
from src.problems.basic_arithmetic.addition import Add
from src.problems.basic_arithmetic.multiplication import Mul
from src.problems.basic_arithmetic.subtraction import Sub
from src.problems.clique import Clique
from src.problems.factorization import Factor
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.tsp import TSP
from src.recommender.recommender_engine import recommender, plot_results
from src.reduction import *
from src.sat_to_qubo import *
from src.circuits_library import *
import csv
import ast
import os, csv, ast
from collections import defaultdict
import signal

class TimeoutErrorInLoop(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutErrorInLoop("Iteration exceeded time limit")

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.parser = Parser(model_path="../parser/saved_models")
        self.clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input data\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
        self.tsp_snippet = "def a(cost_matrix):\n    n = len(cost_matrix)\n    visited = [0]\n    total_cost = 0\n    current = 0\n    while len(visited) < n:\n        next_city = min([city for city in range(n) if city not in visited], key=lambda city: cost_matrix[current][city])\n        total_cost += cost_matrix[current][next_city]\n        visited.append(next_city)\n        current = next_city\n    total_cost += cost_matrix[visited[-1]][0]\n    return total_cost, visited\n\n# Input data\ncost_matrix = [[0, 11, 30], [11, 0, 35], [30, 35, 0]]\ncost, route = a(cost_matrix)\nprint(cost, route)"
        self.kcolor_snippet = "import networkx as nx\n\ndef kcoloring_networkx_dfs(G, k):\n    def dfs(node, colors):\n        if node == len(G.nodes):\n            return colors\n        for color in range(k):\n            if all(colors[neighbor] != color for neighbor in G.neighbors(node)):\n                colors[node] = color\n                result = dfs(node + 1, colors)\n                if result:\n                    return result\n                colors[node] = -1\n        return None\n    return dfs(0, [-1] * len(G.nodes))\n\nG = nx.Graph()\nG.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])\ncolors = kcoloring_networkx_dfs(G, 3)\nprint(colors)"
        self.vc_snippet = "def vertex_cover_greedy(n, edges):\n    cover = set()\n    remaining_edges = set(edges)\n    while remaining_edges:\n        u, v = remaining_edges.pop()\n        cover.add(u)\n        cover.add(v)\n        remaining_edges = {e for e in remaining_edges if u not in e and v not in e}\n    return cover\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3), (3, 4)]\ncover = vertex_cover_greedy(5, edges)\nprint(cover)"



    def evaluation(self):
        # file_path = '../parser/data.csv'
        file_path = '../parser/extra_data.csv'
        first_column = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    first_column.append(row[0])

        num = 0
        mis_code = ""
        # 🔧 Set up LaTeX output directory and tracking
        latex_output_dir = "latex_reports"
        os.makedirs(latex_output_dir, exist_ok=True)
        problem_counts = defaultdict(int)

        for item in first_column:
            try:
                try:
                    clean_code = ast.literal_eval(item)
                except Exception:
                    clean_code = item.encode().decode('unicode_escape')

                problem_type, data = self.parser.parse(clean_code)
                problem = PROBLEMS[problem_type](data)

                # 🔧 Generate unique LaTeX filename
                problem_counts[problem_type] += 1
                count = problem_counts[problem_type]
                suffix = f"{count}" if count > 1 else ""
                filename = f"{problem_type}{suffix}.tex"
                full_path = os.path.join(latex_output_dir, filename)

                # 🔧 Write LaTeX report
                problem.report_latex(full_path)

                if problem_type == 'Unknown':
                    num += 1
                if problem_type == "MIS":
                    self.mis_code = clean_code

                print("→", problem_type, data, problem)

            except Exception as e:
                print("parse failed:", clean_code[:60].replace('\n', ' ') + "...")
                print("error info:", e)

        # print("Number of Unknown problems:", num)

    def test_something(self):
        tag, data = self.parser.parse(self.is_snippet)
        #print(tag, data)
        mis = PROBLEMS[tag](data)
        mis.report_latex()
        # mis.report_3sat()

    def test_is(self):
        tag, data = self.parser.parse(self.is_snippet)
        # print(tag, data)
        mis = PROBLEMS[tag](data)
        mis.recommender_engine()

    def test_maxcut(self):
        tag, data = self.parser.parse(self.maxCut_snippet)
        print(tag, data)
        problem = PROBLEMS[tag](data)
        problem.report_latex()

    def test_clique(self):
        tag, data = self.parser.parse(self.clique_snippet)
        print(tag, data)
        problem = PROBLEMS[tag](data)
        problem.report_latex()

    def test_factorization(self):
        problem = Factor(35)
        problem.report_latex()

    def test_add(self):
        problem = Add(7, 7)
        problem.report_latex()

    def test_multiply(self):
        problem = Mul(-7, 7)
        problem.report_latex()

    def test_subtract(self):
        problem = Sub(4, 6)
        problem.report_latex()

    def test_tsp(self):
        tag, data = self.parser.parse(self.tsp_snippet)
        print(tag, data)
        problem = PROBLEMS[tag](data)
        problem.report_latex()

    def test_kcolor(self):
        tag, data = self.parser.parse(self.kcolor_snippet)
        print(tag, data.G)
        problem = PROBLEMS[tag](data)
        problem.report_latex()

    def test_vc(self):
        tag, data = self.parser.parse(self.vc_snippet)
        print(tag, data.G)
        problem = PROBLEMS[tag](data)
        problem.report_latex()

    def test(self):
        # file_path = '../parser/data.csv'
        file_path = '../parser/data.csv'
        first_column = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    first_column.append(row[0])

        num = 0
        mis_code = ""
        # 🔧 Set up LaTeX output directory and tracking
        latex_output_dir = "latex_reports"
        os.makedirs(latex_output_dir, exist_ok=True)
        problem_counts = defaultdict(int)

        # Set up alarm-based timeout (Unix/macOS)
        signal.signal(signal.SIGALRM, _timeout_handler)
        TIME_LIMIT_SECS = 60  # 1 minute per item

        for item in first_column:
            # Arm the per-iteration timer
            signal.alarm(TIME_LIMIT_SECS)
            try:
                try:
                    clean_code = ast.literal_eval(item)
                except Exception:
                    clean_code = item.encode().decode('unicode_escape')

                problem_type, data = self.parser.parse(clean_code)
                problem = PROBLEMS[problem_type](data)

                # 🔧 Generate unique LaTeX filename
                problem_counts[problem_type] += 1
                count = problem_counts[problem_type]
                suffix = f"{count}" if count > 1 else ""
                filename = f"{problem_type}{suffix}.tex"
                full_path = os.path.join(latex_output_dir, filename)

                # 🔧 Write LaTeX report
                problem.report_latex(output_path=full_path)

                if problem_type == 'Unknown':
                    num += 1
                if problem_type == "MIS":
                    self.mis_code = clean_code

                print("→", problem_type, data, problem)

            except TimeoutErrorInLoop:
                # Skip this item if it ran too long
                snippet = (item[:60] if isinstance(item, str) else str(item)[:60]).replace('\n', ' ')
                print(f"⏱️ Timeout: processing took over {TIME_LIMIT_SECS}s, skipping this item → {snippet}...")
                continue
            except Exception as e:
                snippet = (item[:60] if isinstance(item, str) else str(item)[:60]).replace('\n', ' ')
                print("parse failed:", snippet + "...")
                print("error info:", e)
            finally:
                # Always disarm the timer before the next iteration
                signal.alarm(0)

if __name__ == '__main__':
    unittest.main()

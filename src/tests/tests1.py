import unittest
import ast
import json
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

class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.is_snippet = "def independent_nodes(n, edges):\n    independent_set = set()\n    for node in range(n):\n        if all(neighbor not in independent_set for u, v in edges if u == node for neighbor in [v]):\n            independent_set.add(node)\n    return independent_set\n\n# Input data\nedges = [(0, 1), (0, 2), (1, 2), (1, 3)]\nindependent_set = independent_nodes(2, edges)\nprint(independent_set)"
        self.maxCut_snippet = "def simple_cut_strategy(edges, n):\n    A, B = set(), set()\n    for node in range(n):\n        if len(A) < len(B):\n            A.add(node)\n        else:\n            B.add(node)\n    return sum(1 for u, v in edges if (u in A and v in B)), A, B\n\n# Input data\nedges = [(0, 1), (1, 2), (2, 3)]\ncut_value, A, B = simple_cut_strategy(edges, 4)\nprint(cut_value, A, B)"
        self.parser = Parser(model_path="../parser/saved_models")
        self.clique_snippet = "def compute_clique(nodes, edges):\n    clique = set()\n    for node in nodes:\n        if all((node, neighbor) in edges or (neighbor, node) in edges for neighbor in clique):\n            clique.add(node)\n    return clique\n\n# Input data\nnodes = [0, 1, 2, 3]\nedges = [(0, 1), (0, 2), (1, 2), (2, 3)]\nresult = compute_clique(nodes, edges)\nprint(result)"
        parser = Parser(model_path="../parser/saved_models")

        # file_path = '../parser/data.csv'
        file_path = '../parser/extra_data.csv'
        first_column = []

        # 读取 CSV 文件第一列
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    first_column.append(row[0])

        num = 0  # 统计 'Unknown' 类型的个数
        mis_code = ""
        # 遍历每条记录
        for item in first_column:
            try:
                # 尝试将字符串中的 \n 等还原成真实代码
                try:
                    clean_code = ast.literal_eval(item)
                except Exception:
                    clean_code = item.encode().decode('unicode_escape')

                # 使用 parser 解析 clean_code
                problem_type, data = parser.parse(clean_code)

                # 打印结果
                print("→", problem_type, data,)

                # 统计 Unknown 类型
                if problem_type == 'Unknown':
                    num += 1
                if problem_type == "MIS":
                    self.mis_code = clean_code

            except Exception as e:
                print("parse failed:", clean_code[:60].replace('\n', ' ') + "...")
                print("error info:", e)

        # print("Number of Unknown problems:", num)
    def test_something(self):
        tag, data = self.parser.parse(self.is_snippet)
        #print(tag, data)
        mis = PROBLEMS[tag](data.G)
        mis.report_latex()
        # mis.report_3sat()

    def test_is(self):
        tag, data = self.parser.parse(self.is_snippet)
        # print(tag, data)
        mis = PROBLEMS[tag](data.G)
        mis.recommender_engine()

    def test_maxcut(self):
        tag, data = self.parser.parse(self.maxCut_snippet)
        print(tag, data)
        problem = PROBLEMS[tag](data.G)
        problem.report_latex()

    def test_clique(self):
        tag, data = self.parser.parse(self.clique_snippet)
        print(tag, data)
        problem = PROBLEMS[tag](data.G)
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





if __name__ == '__main__':
    unittest.main()

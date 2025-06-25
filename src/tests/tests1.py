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
        self.parser = Parser(model_path="../../others/saved_models")
        parser = Parser(model_path="../../others/saved_models")

        file_path = '../parser/data.csv'
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
                # print("→", problem_type, data)

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
        mis.report_3sat()



if __name__ == '__main__':
    unittest.main()

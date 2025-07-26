import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from pylatex import Document
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit

from src.problems.qubo import QUBO
import sys
import types

for mod in [
    'qiskit_ionq',
    'pytket',
    'pytket.extensions',
    'pytket.extensions.quantinuum',
    'pytket.extensions.qiskit',
    'braket',
    'braket.aws',
]:
    sys.modules.setdefault(mod, types.ModuleType(mod))

# Provide dummy classes used in recommender_engine imports
class _DummyProvider:
    def get_backend(self, *args, **kwargs):
        class _Backend:
            pass
        return _Backend()

sys.modules['qiskit_ionq'].IonQProvider = _DummyProvider
sys.modules['pytket.extensions.quantinuum'].QuantinuumBackend = object
sys.modules['pytket.extensions.qiskit'].qiskit_to_tk = lambda x: x
sys.modules['pytket.extensions.qiskit'].tk_to_qiskit = lambda x: x
sys.modules['braket.aws'].AwsDevice = object

from src.problems.np_complete import NPC


def dummy_qaoa_optimize(qubo, layers=3):
    qc = QuantumCircuit(len(qubo))
    return {"qc": qc, "parameters": None, "theta": None}


def dummy_vqe_optimization(qubo, layers=3):
    qc = QuantumCircuit(len(qubo))
    return {"qc": qc, "parameters": None, "theta": None}


def dummy_sample_results(*args, **kwargs):
    qc = args[0]
    return [0] * qc.num_qubits


def dummy_recommender(qc, save_figures=True, ibm_service=None, available_devices=None):
    directory = dummy_recommender.output_dir
    if save_figures:
        for name in [
            "recommender_errors_devices.png",
            "recommender_times_devices.png",
            "recommender_prices_devices.png",
        ]:
            path = os.path.join(directory, name)
            plt.figure()
            plt.plot([0, 1], [0, 1])
            plt.savefig(path)
            plt.close()
    return "dummy", []


class DummyProblem(NPC):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.nodes = list(graph.nodes())

    def to_qubo(self):
        n = len(self.nodes)
        return QUBO(np.zeros((n, n)))

    def draw_result(self, result, pos=None):
        plt.figure()
        nx.draw(self.graph, pos=pos, with_labels=True)

    def grover_sat(self, iterations):
        return QuantumCircuit(len(self.nodes))

    def to_sat(self):
        pass


class HelperTests(unittest.TestCase):
    def setUp(self):
        g = nx.Graph([(0, 1), (1, 2)])
        self.problem = DummyProblem(g)
        self.directory = tempfile.mkdtemp()
        self.doc = Document()
        self.pos = nx.spring_layout(g)

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_graph_helper(self):
        path = self.problem._graph_latex(self.doc, self.pos, self.directory)
        self.assertTrue(os.path.exists(path))

    def test_qaoa_helper(self):
        with patch('src.problems.np_complete.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.problems.base.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.algorithms.QAOA.QAOA.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.algorithms.QAOA.QAOA.sample_results', dummy_sample_results):
            res = self.problem._qaoa_latex(self.doc, self.pos, self.directory)
        self.assertTrue(os.path.exists(res['result_path']))
        self.assertTrue(os.path.exists(res['circuit_path']))

    def test_vqe_helper(self):
        with patch('src.problems.np_complete.vqe_optimization', dummy_vqe_optimization), \
             patch('src.problems.base.vqe_optimization', dummy_vqe_optimization), \
             patch('src.algorithms.VQE.VQE.vqe_optimization', dummy_vqe_optimization), \
             patch('src.algorithms.VQE.VQE.sample_results', dummy_sample_results):
            res = self.problem._vqe_latex(self.doc, self.pos, self.directory)
        self.assertTrue(os.path.exists(res['result_path']))
        self.assertTrue(os.path.exists(res['circuit_path']))

    def test_device_recommendation_helper(self):
        with patch('src.problems.np_complete.recommender', dummy_recommender), \
             patch('src.recommender.recommender_engine.recommender', dummy_recommender), \
             patch('src.problems.np_complete.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.problems.base.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.algorithms.QAOA.QAOA.qaoa_optimize', dummy_qaoa_optimize), \
             patch('src.algorithms.QAOA.QAOA.sample_results', dummy_sample_results):
            dummy_recommender.output_dir = self.directory
            qaoa_dict = self.problem._qaoa_latex(self.doc, self.pos, self.directory)
            paths = self.problem._device_recommendation_latex(self.doc, qaoa_dict['qc'], self.directory)
        for p in paths:
            self.assertTrue(os.path.exists(p))


if __name__ == '__main__':
    unittest.main()

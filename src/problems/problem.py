from qiskit import transpile
from qiskit_aer import AerSimulator

from src.algorithms.QAOA.QAOA import qaoa_optimize
from src.algorithms.VQE.VQE import vqe_optimization
from src.graph import Graph
from src.reduction import *
from src.sat_to_qubo import Chancellor
from src.algorithms.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized


class Problem:
    def to_qubo(self):
        raise NotImplementedError("should be implemented in subclass")

    def to_ising(self):
        raise NotImplementedError("should be implemented in subclass")

    def qaoa(self):
        qubo = self.to_qubo().Q
        # Run QAOA on local simulator
        qaoa_dict = qaoa_optimize(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = qaoa_dict["qc"]
        parameters = qaoa_dict["parameters"]
        theta = qaoa_dict["theta"]

        return qc

    def vqe(self):
        qubo = self.to_qubo().Q
        # Run QAOA on local simulator
        vqe_dict = vqe_optimization(qubo, layers=1)

        # Obtain the parameters of the QAOA run
        qc = vqe_dict["qc"]
        parameters = vqe_dict["parameters"]
        theta = vqe_dict["theta"]

        return qc

    def interpret(self, result):
        raise NotImplementedError("should be implemented in subclass")

    def draw_result(self):
        raise NotImplementedError("should be implemented in subclass")

    def grover(self):
        raise NotImplementedError("should be implemented in subclass")

    def dwave_qa(self):
        raise NotImplementedError("should be implemented in subclass")

    def report(self):
        raise NotImplementedError("should be implemented in subclass")

    def report_latex(self):
        raise NotImplementedError("should be implemented in subclass")

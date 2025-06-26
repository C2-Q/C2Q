from src.algorithms.QAOA.QAOA import qaoa_optimize
from src.algorithms.VQE.VQE import vqe_optimization
from src.graph import Graph
from src.reduction import *
from src.sat_to_qubo import Chancellor
from src.algorithms.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized


class Base:
    def report(self):
        raise NotImplementedError("should be implemented in subclass")

    def report_latex(self):
        raise NotImplementedError("should be implemented in subclass")
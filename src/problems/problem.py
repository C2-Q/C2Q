from qiskit import transpile
from qiskit_aer import AerSimulator
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
        raise NotImplementedError("should be implemented in subclass")
    def vqe(self):
        raise NotImplementedError("should be implemented in subclass")
    def grover(self):
        raise NotImplementedError("should be implemented in subclass")
    def dwave_qa(self):
        raise NotImplementedError("should be implemented in subclass")
    def report(self, args: str):
        raise NotImplementedError("should be implemented in subclass")

from qiskit import transpile
from qiskit_aer import AerSimulator
from src.graph import Graph
from src.reducer import *
from src.sat_to_qubo import Chancellor
from src.grover import grover
from src.circuits_library import cnf_to_quantum_oracle_optimized

class Problem:
    def qaoa(self):
        raise NotImplementedError("should be implemented in subclass")

    def vqe(self):
        raise NotImplementedError("should be implemented in subclass")

    def grover(self):
        raise NotImplementedError("should be implemented in subclass")

    def dwave(self):
        raise NotImplementedError("should be implemented in subclass")

    def report(self, args:str):
        raise NotImplementedError("should be implemented in subclass")

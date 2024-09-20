from pysat.formula import CNF

from src.problems.problems import *


class NP(Problem):

    def __init__(self):
        self.sat = None
        self.three_sat = None

    def reduce_to_3sat(self):
        self.three_sat = sat_to_3sat(self.sat)
        chancellor = Chancellor(self.three_sat)
        chancellor.fillQ()
        chancellor.visualizeQ()

    def grover(self, iterations=3):
        oracle = cnf_to_quantum_oracle_optimized(self.sat)
        grover_circuit = grover(oracle,
                                objective_qubits=list(range(self.sat.nv)),
                                iterations=iterations)
        backend = AerSimulator()
        transpiled_circuit = transpile(grover_circuit, backend=backend)
        counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
        return counts

    def qaoa_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

    def grover_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

    def vqe_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

    def dwave_3sat(self):
        raise NotImplementedError("should be implemented in subclass")
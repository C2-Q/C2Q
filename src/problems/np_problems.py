from pysat.formula import CNF
from src.problems.problem import *


def _grover(sat, iterations=3):
    oracle = cnf_to_quantum_oracle_optimized(sat)
    grover_circuit = grover(oracle,
                            objective_qubits=list(range(sat.nv)),
                            iterations=iterations)
    backend = AerSimulator()
    transpiled_circuit = transpile(grover_circuit, backend=backend)
    counts = backend.run(transpiled_circuit, shots=50000).result().get_counts()
    return counts


class NP(Problem):

    def __init__(self):
        self.sat = None
        self.three_sat = None

    def reduce_to_sat(self):
        # should be implemented by specific subclass
        raise NotImplementedError("should be implemented in subclass")

    def reduce_to_3sat(self):
        # if sat is not none
        self.three_sat = sat_to_3sat(self.sat)
        # further reduce literals, <= 3
        chancellor = Chancellor(self.three_sat)
        chancellor.fillQ()
        chancellor.visualizeQ()

    def grover(self, iterations=3):
        return _grover(self.sat, iterations)

    def grover_3sat(self, iterations=3):
        return _grover(self.three_sat, iterations)

    def qaoa_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

    def vqe_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

    def dwave_3sat(self):
        raise NotImplementedError("should be implemented in subclass")

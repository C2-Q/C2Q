from src.algorithms.grover import grover
from src.circuits_library import quantum_factor_mul_oracle
from src.problems.problem import Problem
from qiskit_algorithms import AmplificationProblem

from qiskit.circuit.library import PhaseOracle, GroverOperator
from qiskit_algorithms import AmplificationProblem, Grover
from qiskit import qasm2, QuantumCircuit
from qiskit.primitives import Sampler


# src/algorithms/base_algorithm.py
class BaseAlgorithm:
    def __init__(self):
        self.circuit = None

    def run(self):
        """
        run on simulator
        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method")

    def run_on_quantum(self):
        """
        run on quantum computer
        Returns
        -------

        """
        raise NotImplementedError("Subclasses should implement this method")

    def export_to_qasm(self):
        raise NotImplementedError("Subclasses should implement this method")


class GroverWrapper(BaseAlgorithm):
    def __init__(self,
                 oracle: QuantumCircuit,
                 iterations,
                 is_good_state=None,
                 state_preparation: QuantumCircuit = None,
                 objective_qubits=None
                 ):
        super().__init__()
        self.grover_op = GroverOperator(oracle,
                                        reflection_qubits=objective_qubits)
        if objective_qubits is None:
            objective_qubits = list(range(oracle.num_qubits))
        if state_preparation is None:
            state_preparation = QuantumCircuit(oracle.num_qubits)
            state_preparation.h(objective_qubits)
        if is_good_state is None:
            def func(state):
                return True

            is_good_state = func
        self.circuit = QuantumCircuit(oracle.num_qubits, len(objective_qubits))
        self.circuit.compose(state_preparation, inplace=True)
        self.circuit.compose(self.grover_op.power(iterations),
                             inplace=True)
        self.circuit.measure(objective_qubits,
                             objective_qubits)
        self.problem = AmplificationProblem(oracle,
                                            state_preparation=state_preparation,
                                            is_good_state=is_good_state,
                                            objective_qubits=objective_qubits)
        self.grover = Grover(sampler=Sampler(), iterations=iterations)

    def run(self, verbose=False):
        result = self.grover.amplify(self.problem)
        if verbose:
            print(result)
        return result

    def run_on_quantum(self):
        return None

    def export_to_qasm(self):
        if self.circuit is None:
            raise ValueError("Grover operator has not been generated yet. Call generate_quantum_code() first.")
        qasm = self.circuit.qasm()
        #TODO error with c3sqrtx
        #qasm = qasm.replace("c3sqrtx", "c3sx")
        return qasm


class Factor(Problem):
    def __init__(self, number):
        self.number = number

    def grover(self, iterations=2):
        oracle, prep_state, obj_bits, working_bits = quantum_factor_mul_oracle(self.number)
        print(list(range(prep_state.num_qubits)))
        grover_circuit = grover(oracle,
                                objective_qubits=obj_bits,
                                iterations=iterations,
                                working_qubits=working_bits,
                                state_pre=prep_state,
                                )
        return grover_circuit

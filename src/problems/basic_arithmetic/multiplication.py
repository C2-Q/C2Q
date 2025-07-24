from collections import Counter

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import VBERippleCarryAdder, RGQFTMultiplier

from src.problems.basic_arithmetic.arithmetic import Arithmetic
from src.problems.basic_arithmetic.utils import decimal_to_complement_binary_list, complement_binary_list_to_decimal


class Mul(Arithmetic):
    def __init__(self, left, right):
        super().__init__(left, right)

    def quantum_circuit(self):
        """
        Perform the multiplication of the two operands (left and right)
        using quantum circuit (measure gates added!!!)
        for the time being, took qiskit implementation of QFT as a reference to make
        it work
        :return: The result of left * right
                and corresponding circuit
        """
        left = self.left
        right = self.right
        num_state_qubits = max(left.bit_length(), right.bit_length())
        num_result_qubits = num_state_qubits * 2
        q = QuantumRegister(num_state_qubits * 2 + num_result_qubits, 'q')
        c = ClassicalRegister(num_result_qubits, 'c')
        qc = QuantumCircuit(q, c)

        # Set the bits for Operand A (from right to left)
        for i in range(left.bit_length()):
            if (left >> i) & 1:
                qc.x(q[i])

        # Set the bits for Operand B (from right to left)
        for i in range(right.bit_length()):
            if (right >> i) & 1:
                qc.x(q[i + num_state_qubits])

        multiplier_circuit = RGQFTMultiplier(num_state_qubits=num_state_qubits, num_result_qubits=num_result_qubits)
        qc = qc.compose(multiplier_circuit)

        for i in range(num_result_qubits):
            qc.measure(q[i + num_state_qubits * 2], c[i])

        # sampler = Sampler()
        # result = sampler.run(circuit).result()
        # result_counts = result.quasi_dists[0]
        # result_value = max(result_counts, key=result_counts.get)

        return qc

    def interpret(self, result):
        result = Counter(result).most_common(1)
        result = result[0][0]
        result = int(result.replace(' ', ''), 2)
        n_bits = max(self.left.bit_length(), self.right.bit_length()) * 2
        if self.left * self.right < 0:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        else:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        return result


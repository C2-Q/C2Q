from qiskit import QuantumCircuit

from src.problems.basic_arithmetic.arithmetic import Arithmetic
from src.problems.basic_arithmetic.utils import decimal_to_complement_binary_list, complement_binary_list_to_decimal


class Add(Arithmetic):
    def __init__(self, left, right):
        super().__init__(left, right)

    def quantum_circuit(self):
        """
        Perform the addition of the two operands (left and right)
        using quantum circuit (measure gates added!!!)
        :return: The result of left + right
                and corresponding circuit
        """
        left = self.left
        right = self.right
        n_bits = max(left.bit_length(), right.bit_length()) + 1
        # Ensure both left and right have n_bits length
        left_list = decimal_to_complement_binary_list(left, n_bits)
        right_list = decimal_to_complement_binary_list(right, n_bits)

        # Create a quantum circuit with 2*n_bits for input and 1 additional for carry
        if left * right > 0:
            qc = QuantumCircuit(3 * n_bits + 1, n_bits + 1)
        else:
            qc = QuantumCircuit(3 * n_bits + 1, n_bits)
        # Initialize the input qubits
        for i in range(n_bits):
            if left_list[i] == 1:
                qc.x(i)  # Set qubit for left bit
            if right_list[i] == 1:
                qc.x(n_bits + i)  # Set qubit for right bit

        # Apply quantum gates for addition
        for i in range(n_bits):
            qc.ccx(i, n_bits + i, 2 * n_bits + i + 1)
            qc.cx(i, n_bits + i)
            qc.ccx(n_bits + i, 2 * n_bits + i, 2 * n_bits + i + 1)
            qc.cx(n_bits + i, 2 * n_bits + i)
            qc.cx(i, n_bits + i)

        # Measuring the result
        for i in range(n_bits):
            qc.measure(i + 2 * n_bits, i)  # Measure left bits to result bits
        if left * right > 0:
            qc.measure(3 * n_bits, n_bits)
        # execution required from recommender side
        # sampler = Sampler()
        # result = sampler.run(circuits=qc, shots=1024).result()
        # result = list(result.quasi_dists[0].keys())[0]
        # if left * right > 0:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        # else:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return qc

    def interpret(self, result):
        n_bits = max(self.left.bit_length(), self.right.bit_length())
        if self.left * self.right > 0:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        else:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return result


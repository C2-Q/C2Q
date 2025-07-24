from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import VBERippleCarryAdder

from src.problems.basic_arithmetic.arithmetic import Arithmetic
from src.problems.basic_arithmetic.utils import decimal_to_complement_binary_list, complement_binary_list_to_decimal


class Sub(Arithmetic):
    def __init__(self, left, right):
        super().__init__(left, right)

    def quantum_circuit(self):
        """
        Perform the subtraction of the two operands (left and right)
        using quantum circuit (measure gates added!!!)
        for the time being, took qiskit implementation as a reference to make
        it work
        :return: The result of left - right
                and corresponding circuit
        """
        left = self.left
        right = self.right
        n_bits = max(left.bit_length(), right.bit_length())
        # Ensure both minuend and subtrahend have n_bits length
        minuend = decimal_to_complement_binary_list(left, n_bits)
        subtrahend = decimal_to_complement_binary_list(right, n_bits)

        # Create a quantum circuit with 2*n_bits for input and 1 additional for borrow
        adder = VBERippleCarryAdder(num_state_qubits=n_bits)
        num_qubits = len(adder.qubits)
        if left * right < 0:
            qc = QuantumCircuit(num_qubits, n_bits + 1)
        else:
            qc = QuantumCircuit(num_qubits, n_bits)
        # Initialize the input qubits
        for i in range(n_bits):
            if minuend[i] == 1:
                qc.x(i + 1)  # Set qubit for minuend bit
            if subtrahend[i] == 1:
                qc.x(i + 1 + n_bits)  # Set qubit for subtrahend bit

        qc.barrier()
        qc.x(range(n_bits + 1, 2 * n_bits + 1))

        qc.x(0)

        qc.append(adder, range(num_qubits))
        for i in range(n_bits):
            qc.measure(i + n_bits + 1, i)
        if left * right < 0:
            qc.measure(n_bits + n_bits + 1, n_bits)
        # sampler = Sampler()
        # result = sampler.run(circuits=qc, shots=1024).result()
        # result = list(result.quasi_dists[0].keys())[0]
        # if left * right < 0:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        # else:
        #     result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return qc

    def interpret(self, result):
        result = Counter(result).most_common(1)
        result = result[0][0]
        result = int(result.replace(' ', ''), 2)
        n_bits = max(self.left.bit_length(), self.right.bit_length())
        if self.left * self.right < 0:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits + 1))
        else:
            result = complement_binary_list_to_decimal(decimal_to_complement_binary_list(result, n_bits))
        return result


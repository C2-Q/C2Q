from itertools import combinations

import networkx as nx
import qiskit.qasm2
from matplotlib import pyplot as plt
from pysat.formula import CNF
from qiskit.circuit.library import PhaseOracle

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import matplotlib.pyplot as plt
# Example CNF Formula using PySAT
from qiskit.circuit.library import OR


def cnf_to_quantum_circuit_optimized(cnf_formula):
    """
    Converts a CNF formula into a quantum circuit with optimized logic and correct carry management.

    Parameters
    ----------
    cnf_formula: A CNF formula object
        The CNF formula to convert into a quantum circuit.
        The CNF formula consists of a list of clauses, where each clause is a list of literals.
        A literal is a positive or negative integer representing a variable or its negation.

    Returns
    -------
    qc : QuantumCircuit
        The quantum circuit that represents the CNF formula.
    """
    num_vars = cnf_formula.nv  # Number of variables
    num_clauses = len(cnf_formula.clauses)

    # Number of qubits required to count up to the number of clauses
    num_count_qubits = (num_clauses).bit_length()  # log2(num_clauses + 1) rounded up

    # Create a quantum circuit with input qubits, count ancillas, and 1 final output qubit
    qc = QuantumCircuit(num_vars + num_count_qubits + 1 + 1)  # +1 for the final output qubit + 1 for the final result

    # Ancilla qubits to store the count of satisfied clauses
    count_ancillas = list(range(num_vars, num_vars + num_count_qubits))
    final_ancilla = num_vars + num_count_qubits

    final_result = final_ancilla + 1
    # Initialize the final ancilla qubit to |1⟩ (assume all clauses are satisfied)
    #qc.x(final_ancilla)

    for i, clause in enumerate(cnf_formula.clauses):
        clause_qubits = []
        clause_negations = []

        # Step through each literal in the clause
        for lit in clause:
            qubit_index = abs(lit) - 1  # Convert variable index to qubit index
            if lit < 0:  # If the literal is negative (~x)
                qc.x(qubit_index)  # Apply X to flip qubit before using in OR
                clause_negations.append(qubit_index)
            clause_qubits.append(qubit_index)

        # OR gate for the clause, outputting to a temporary ancilla qubit
        clause_ancilla = num_vars + num_count_qubits
        or_gate = OR(len(clause_qubits))
        qc.append(or_gate, clause_qubits + [clause_ancilla])

        # Increment logic with proper carry management
        # Start with the least significant bit
        # qc.cx(clause_ancilla, count_ancillas[0])
        # For subsequent bits, handle the carry propagation
        for j in range(1, num_count_qubits):
            control_qubits = [clause_ancilla]
            for i in range(len(count_ancillas) - j):
                control_qubits.append(count_ancillas[i])
            qc.mcx(control_qubits, count_ancillas[-j])
            # Propagate the carry from the lower bits
            #qc.mcx()
            #qc.ccx(clause_ancilla, count_ancillas[j - 1], count_ancillas[j])
        qc.cx(clause_ancilla, count_ancillas[0])
        # Uncompute the OR gate
        qc.append(or_gate.inverse(), clause_qubits + [clause_ancilla])

        # Revert any negations
        if clause_negations:
            qc.x(clause_negations)
        qc.barrier()

    # Final comparison: Check if the count matches the total number of clauses
    # Convert the number of clauses to binary and compare with the ancillas
    control_qubits = []
    for j in range(num_count_qubits):
        if (num_clauses >> j) & 1:
            control_qubits.append(count_ancillas[j])
            #qc.cx(count_ancillas[j], final_ancilla)
    #print(count_ancillas)
    qc.mcx(control_qubits, final_ancilla)
    qc.cx(final_ancilla, final_result)
    qc.reset(count_ancillas)
    qc.reset(final_ancilla)
    # If the count matches the number of clauses, final_ancilla will remain |1⟩
    # If not, it will be flipped to |0⟩
    return qc


def cnf_to_quantum_oracle_optimized(cnf_formula):
    qc = cnf_to_quantum_circuit_optimized(cnf_formula)
    # Create an empty quantum circuit to hold the full oracle operations
    qc_tmp = QuantumCircuit(qc.num_qubits)

    qc_tmp.barrier()
    # Prepare the final ancilla qubit in the |-> state
    qc_tmp.x(qc.num_qubits - 1)
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.barrier()

    # Append the CNF quantum circuit (the oracle construction)
    qc_tmp.compose(qc, inplace=True)

    # Flip the phase of the ancilla qubit back
    qc_tmp.h(qc.num_qubits - 1)
    qc_tmp.x(qc.num_qubits - 1)

    qc_tmp.barrier()

    return qc_tmp


def clique_oracle(graph: nx.Graph, k):
    """
    :param graph:
    :param k:
    :return:
    """

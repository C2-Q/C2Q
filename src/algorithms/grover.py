import numpy
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, MCXGate
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def grover(oracle: QuantumCircuit,
           objective_qubits=None,
           iterations=2):
    """
    Implements Grover's algorithm on a given oracle.
    If no objective qubits are provided, all qubits of the oracle will be used.

    Parameters:
    - oracle (QuantumCircuit): A quantum circuit that implements the oracle function.
                               The oracle should flip the sign of the desired solution states.
    - objective_qubits (list, optional): A list of qubits that represent the objective function.
                                         If None, all qubits from the oracle are used.
    - iterations (int): The number of Grover iterations to apply.

    Returns:
    - grover_circuit (QuantumCircuit): The complete Grover circuit ready for execution.
    """
    # Determine the number of qubits from the oracle
    num_qubits = oracle.num_qubits
    # If no objective qubits are given, use all qubits
    if objective_qubits is None:
        objective_qubits = list(range(num_qubits))

    # Create registers and the quantum circuit
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(len(objective_qubits))
    grover_circuit = QuantumCircuit(qr, cr)

    # Initialize all qubits to the uniform superposition state |+>
    grover_circuit.h(objective_qubits)

    # Apply the Grover iterations
    # Grover operator iterations times
    for _ in range(iterations):
        # Apply the oracle
        oracle.name = "oracle"
        grover_circuit.append(oracle, qr)

        # Apply the Grover diffusion operator using all specified objective qubits
        grover_circuit.h(objective_qubits)
        grover_circuit.x(objective_qubits)

        # Apply a multi-controlled Z gate to reflect the |111...1> state
        grover_circuit.h(objective_qubits[-1])
        grover_circuit.mcx(objective_qubits[:-1], objective_qubits[-1])  # Apply multi-controlled-X
        grover_circuit.h(objective_qubits[-1])

        grover_circuit.x(objective_qubits)
        grover_circuit.h(objective_qubits)
        # end of diffuser

    grover_circuit.global_phase = numpy.pi
    # Measure all qubits in the circuit
    grover_circuit.measure(objective_qubits, cr)

    return grover_circuit

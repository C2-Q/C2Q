import numpy
import numpy as np
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, MCXGate
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.visualization import plot_distribution
import pickle


def grover(oracle: QuantumCircuit,
           state_pre: QuantumCircuit = None,
           objective_qubits=None,
           working_qubits=None,
           iterations=2):
    """
    Implements Grover's algorithm on a given oracle.
    If no objective qubits are provided, all qubits of the oracle will be used.
    Default state preparation
    Parameters:
    - oracle (QuantumCircuit): A quantum circuit that implements the oracle function.
                               The oracle should flip the sign of the desired solution states.
    - state_prep (QuantumCircuit): A quantum circuit that implements the state preparation at the beginning of the
                                circuit and the beginning and end of diffuser.
    - objective_qubits (list, optional): A list of qubits that represent the objective function.
                                         If None, all qubits from the oracle are used.
    - working_qubits (list, optional): Actual working qubits of oracle, opposite to ancillary qubits in oracle
    - iterations (int): The number of Grover iterations to apply.

    Returns:
    - grover_circuit (QuantumCircuit): The complete Grover circuit ready for execution.
    """
    # Determine the number of qubits from the oracle
    num_qubits = oracle.num_qubits
    # If no objective qubits are given, use all qubits
    if objective_qubits is None:
        objective_qubits = list(range(num_qubits))
    if working_qubits is None:
        working_qubits = list(range(num_qubits))
    if state_pre is None:
        state_pre = QuantumCircuit(num_qubits)
        state_pre.h(list(range(num_qubits)))
    # Create registers and the quantum circuit
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(len(objective_qubits))
    grover_circuit = QuantumCircuit(qr, cr)

    # Initialize all qubits to the uniform superposition state |+>
    grover_circuit = grover_circuit.compose(state_pre)
    # Apply the Grover iterations
    # Grover operator iterates times
    for _ in range(iterations):
        # Apply the oracle
        oracle.name = "oracle"
        grover_circuit.append(oracle, qr)

        # Apply the Grover diffusion operator using all specified objective qubits
        grover_circuit.h(working_qubits)
        grover_circuit.x(working_qubits)

        # Apply a multi-controlled Z gate to reflect the |111...1> state
        grover_circuit.h(working_qubits[-1])
        grover_circuit.mcx(working_qubits[:-1], working_qubits[-1])  # Apply multi-controlled-X
        grover_circuit.h(working_qubits[-1])

        grover_circuit.x(working_qubits)
        grover_circuit.h(working_qubits)
        # end of diffuser

    grover_circuit.global_phase = numpy.pi
    # Measure all qubits in the circuit
    grover_circuit.measure(objective_qubits, cr)

    return grover_circuit


def grover_optimized_iterations(oracle: QuantumCircuit,
                                state_pre: QuantumCircuit = None,
                                objective_qubits=None,
                                working_qubits=None,
                                ):
    """
    Implements Grover's algorithm on a given oracle.

    Returns:
    - grover_circuit (List of QuantumCircuit): The complete Grover circuit ready for execution.
    """
    # Determine the number of qubits from the oracle
    num_qubits = oracle.num_qubits
    # If no objective qubits are given, use all qubits
    if objective_qubits is None:
        objective_qubits = list(range(num_qubits))
    if working_qubits is None:
        working_qubits = list(range(num_qubits))
    if state_pre is None:
        state_pre = QuantumCircuit(num_qubits)
        state_pre.h(list(range(num_qubits)))
    # Create registers and the quantum circuit
    qr = QuantumRegister(num_qubits)
    cr = ClassicalRegister(len(objective_qubits))
    grover_circuit = QuantumCircuit(qr, cr)

    # Initialize all qubits to the uniform superposition state |+>
    grover_circuit = grover_circuit.compose(state_pre)
    # Apply the Grover iterations
    # Grover operator iterates times
    # Start from iterations = 1
    # todo
    for _ in range(iterations):
        # Apply the oracle
        oracle.name = "oracle"
        grover_circuit.append(oracle, qr)

        # Apply the Grover diffusion operator using all specified objective qubits
        grover_circuit.h(working_qubits)
        grover_circuit.x(working_qubits)

        # Apply a multi-controlled Z gate to reflect the |111...1> state
        grover_circuit.h(working_qubits[-1])
        grover_circuit.mcx(working_qubits[:-1], working_qubits[-1])  # Apply multi-controlled-X
        grover_circuit.h(working_qubits[-1])

        grover_circuit.x(working_qubits)
        grover_circuit.h(working_qubits)
        # end of diffuser

    grover_circuit.global_phase = numpy.pi
    # Measure all qubits in the circuit
    grover_circuit.measure(objective_qubits, cr)

    return grover_circuit

def sample_results(grover_circuit: QuantumCircuit, backend=AerSimulator()):
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1000
    transpiled_circuit = transpile(grover_circuit, backend, seed_transpiler=77, layout_method='sabre', routing_method='sabre')
    job = sampler.run([transpiled_circuit])
    result = job.result()[0]
    counts = result.data.c0.get_counts()

    most_probable_grover_result = max(counts, key=counts.get)
    most_probable_grover_result = np.fromstring(most_probable_grover_result, np.int8) - 48
    # Flip the bitstring to fix the order
    # from intrinsic little endian to big endian
    most_probable_grover_result = most_probable_grover_result[::-1]
    return most_probable_grover_result
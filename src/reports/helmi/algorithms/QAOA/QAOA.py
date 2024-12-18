from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit.visualization import plot_distribution

from scipy.optimize import minimize

import numpy as np
import pickle

def convert_qubo_to_ising(qubo):
    # Number of qubits
    n = len(qubo)

    # Calculate the offset also
    offset = 0

    operator_list = []

    for i in range(n):
        for j in range(i, n):
            # Initialize the Pauli operator with all I's
            pauli_operator = list("I" * n)

            # Use only the upper triangular part of the matrix
            if j >= i:
                if i == j:
                    pauli_operator[i] = "Z"
                    ising_value = -(1/2)*qubo[i][i] - (1/4)*np.sum(qubo[i][(i+1):]) - (1/4)*np.sum(qubo[:,i][:i])
                    offset += (1 / 2) * qubo[i][i]
                else:
                    pauli_operator[i] = "Z"
                    pauli_operator[j] = "Z"
                    ising_value = (1 / 4) * qubo[i][j]
                    offset += (1 / 4) * qubo[i][j]

                if not ising_value == 0:
                    ising_pauli_op = (''.join(pauli_operator), ising_value)
                    operator_list.append(ising_pauli_op)

    operators = SparsePauliOp.from_list(operator_list)

    return operators, offset

def initialize_qaoa(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()

    return qc

# Add a cost layer based on the Ising Hamiltonian
def add_cost_layer(qc, ising, gamma, n):
    cost_layer = PauliEvolutionGate(ising, gamma)
    qc.append(cost_layer, range(n))
    qc.barrier()

# Add a QAOA mixer layer
def add_mixer_layer(qc, beta, n):
    qc.rx(2 * beta, range(n))
    #qc.barrier()

def add_qaoa_layer(qc, ising, parameters, layers, n):
    i = 0
    while i < layers * 2:
        # Apply cost layer
        add_cost_layer(qc, ising, parameters[i], n)

        # Apply mixer layer
        add_mixer_layer(qc, parameters[i + 1], n)

        # Move to next QAOA layer
        i += 2

def initialize_parameters(layers):
    theta = []

    # Initialize a parameter for the "gamma" and "beta" variables
    initial_gamma = 2 * np.pi
    initial_beta = np.pi

    initial_param_list = [initial_gamma, initial_beta] * layers
    theta.extend(initial_param_list)

    return theta

def calculate_expectation_value(theta, qc_transpiled, parameters, qubo, backend, exp_value_list, shots=1000):
    # Assign the gammas and the betas to the circuit
    qc_transpiled_assigned_parameters = qc_transpiled.assign_parameters({parameters:theta})

    # Run the circuit on the backend
    result = backend.run(qc_transpiled_assigned_parameters, shots=shots).result()

    # Save the counts to a variable
    counts = result.get_counts()

    exp_value = 0

    for bitstring in counts.keys():
        # Calculate the occurrence of a bitstring with respect to the number of shots
        occurrence_bitstring = counts[bitstring] / shots

        # Calculate the QUBO result of a bitstring
        qubo_value = 0
        for i in range(len(qubo)):
            for j in range(len(qubo)):
                if i == j:
                    qubo_value += qubo[i][j]*int(bitstring[i])
                else:
                    qubo_value += qubo[i][j]*int(bitstring[i])*int(bitstring[j])

        # Expectation values of all bitstrings are summed
        exp_value += occurrence_bitstring * qubo_value
    
    exp_value_list.append(exp_value)

    #print(exp_value)
    
    return exp_value

def optimize_parameters(qc_transpiled, qubo, parameters, theta, backend):
    # Save the expectation values the optimization gives us so that we can visualize the optimization
    exp_value_list = []

    # Here we can change the optimization method etc.
    min_minimized_optimization = minimize(calculate_expectation_value, theta, method="Powell", options={'maxiter':100, 'maxfev':100},
                                          args=(qc_transpiled, parameters, qubo, backend, exp_value_list))

    # Save the objective value the optimization finally gives us
    minimum_objective_value = min_minimized_optimization.fun
    min_exp_value_list = exp_value_list

    return min_minimized_optimization.x, minimum_objective_value, min_exp_value_list

def qaoa_no_optimization(qubo, layers):
    """
    Implements QAOA with given QUBO without optimization. Circuit can be optimized later.

    Parameters:
    - qubo (numpy.ndarray): A QUBO matrix which defines the problem to be solved.
    - layers (int): The number of QAOA layers to apply.

    Returns:
    - qc (QuantumCircuit): Complete QAOA circuit.
    - parameters (ParameterVector): A list of parameters used in the QAOA circuit.
    - theta (numpy.ndarray): An array of initial parameters for the QAOA
    """

    # Number of qubits = length of the QUBO matrix
    n = len(qubo)

    # Initialize circuit
    qc = initialize_qaoa(n)

    # Initialize parameters
    parameters = ParameterVector('theta', 2 * layers)
    theta = initialize_parameters(layers)

    # Convert the QUBO matrix to the Ising Hamiltonian
    ising, offset = convert_qubo_to_ising(qubo)

    # Apply the QAOA layers
    add_qaoa_layer(qc, ising, parameters, layers, n)

    # Add measurements for accurate gate numbers for the recommender system
    qc.measure_all()

    qaoa_dict = {
        "qc": qc,
        "parameters": parameters,
        "theta": theta
    }

    # Return QAOA circuit, parameter list and initial values for the parameters
    return qaoa_dict

def qaoa_optimize(qubo, layers, backend=AerSimulator()):
    """
    Implements QAOA with given QUBO.

    Parameters:
    - qubo (numpy.ndarray): A QUBO matrix which defines the problem to be solved.
    - layers (int): The number of QAOA layers to apply.
    - backend: The backend to run the QAOA on.

    Returns:
    - qc (QuantumCircuit): Complete QAOA circuit.
    - parameters (ParameterVector): A list of parameters used in the QAOA circuit.
    - theta (numpy.ndarray): An array of optimized parameters
    - minimum_objective_value (float): Minimum objective value at the end of the optimization
    - exp_value_list (list): A list of expectation values in every QAOA layer
    """

    # Number of qubits = length of the QUBO matrix
    n = len(qubo)

    # Initialize circuit
    qc = initialize_qaoa(n)

    # Initialize parameters
    parameters = ParameterVector('theta', 2 * layers)
    theta = initialize_parameters(layers)

    # Convert the QUBO matrix to the Ising Hamiltonian
    ising, offset = convert_qubo_to_ising(qubo)

    # Apply the QAOA layers
    add_qaoa_layer(qc, ising, parameters, layers, n)

    qc.measure_all()
    qc_transpiled = transpile(qc, backend, seed_transpiler=77, layout_method='sabre', routing_method='sabre')

    #estimator = Estimator(backend)
    #estimator.options.default_shots = 1000

    # Optimize the parameters
    #theta, minimum_objective_value, exp_value_list = optimize_parameters(qc, qubo, parameters, theta, backend)
    theta, minimum_objective_value, exp_value_list = optimize_parameters(qc_transpiled, qubo, parameters, theta, backend)

    qaoa_dict = {
        "qc": qc_transpiled,
        "parameters": parameters,
        "theta": theta,
        "minimum_objective_value": minimum_objective_value,
        "exp_value_list": exp_value_list,
        "offset": offset
    }

    qaoa_dict_saved = qaoa_dict.copy()
    del qaoa_dict_saved["qc"]
    del qaoa_dict_saved["parameters"]

    with open('qaoa_dict_saved.pkl', 'wb') as f:
        pickle.dump(qaoa_dict_saved, f)

    # Return QAOA circuit, parameter list, optimized values for the parameters, minimum objective value at the end of the optimization and expectation values (objective values) in every QAOA layer
    return qaoa_dict

def sample_results(qc_transpiled, parameters, theta, backend=AerSimulator()):
    if backend.name == 'aer_simulator':
        qc_transpiled_parameters = qc_transpiled.decompose(reps=1).assign_parameters({parameters: theta})
    else:
        qc_transpiled_parameters = qc_transpiled.assign_parameters({parameters: theta})

    counts = backend.run(qc_transpiled_parameters, shots=1000).result().get_counts()

    with open('qaoa_counts.pkl', 'wb') as f:
        pickle.dump(counts, f)

    plot_distribution(counts, figsize=(15, 5), filename='qaoa_distribution.png')

    highest_possible_solution = 0
    max_count = 0
    for key, count in counts.items():
        if count > max_count:
            max_count = count
            highest_possible_solution = key

    # Convert string to array
    X = np.fromiter(highest_possible_solution, dtype=int)

    #print(f'Most probable solution: {highest_possible_solution}')
    return X
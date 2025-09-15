from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler

from scipy.optimize import minimize

import numpy as np

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

def cost_estimator(theta, qc_transpiled, ising, estimator, exp_value_list):
    # Calculate the expectation value

    isa_hamiltonian = ising.apply_layout(qc_transpiled.layout)
    job = estimator.run([(qc_transpiled, isa_hamiltonian, theta)])

    result = job.result()[0]
    cost = result.data.evs

    exp_value_list.append(cost)

    return cost

def optimize_parameters(qc_transpiled, ising, parameters, theta, estimator):
    # Save the expectation values the optimization gives us so that we can visualize the optimization
    exp_value_list = []

    # Here we can change the optimization method etc.
    # SPSA by default, but here we use powell optimizer for testing...
    min_minimized_optimization = minimize(cost_estimator, theta, method="Powell", options={'maxiter':500, 'maxfev':500},
                                          args=(qc_transpiled, ising, estimator, exp_value_list))

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

    estimator = Estimator(backend)
    estimator.options.default_shots = 1000

    # Optimize the parameters
    #theta, minimum_objective_value, exp_value_list = optimize_parameters(qc, qubo, parameters, theta, backend)
    theta, minimum_objective_value, exp_value_list = optimize_parameters(qc_transpiled, ising, parameters, theta, estimator)

    qaoa_dict = {
        "qc": qc,
        "parameters": parameters,
        "theta": theta,
        "minimum_objective_value": minimum_objective_value,
        "exp_value_list": exp_value_list,
        "offset": offset
    }

    # Return QAOA circuit, parameter list, optimized values for the parameters, minimum objective value at the end of the optimization and expectation values (objective values) in every QAOA layer
    return qaoa_dict

def sample_results(qc, parameters, theta, backend=AerSimulator()):
    qc_transpiled = transpile(qc, backend, seed_transpiler=77, layout_method='sabre', routing_method='sabre')

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1000

    if backend.name == 'aer_simulator':
        qc_transpiled_parameters = qc_transpiled.decompose(reps=1).assign_parameters({parameters: theta})
    else:
        qc_transpiled_parameters = qc_transpiled.assign_parameters({parameters: theta})

    job = sampler.run([qc_transpiled_parameters])

    result = job.result()[0]

    counts = result.data.meas.get_counts()

    highest_possible_solution = 0
    max_count = 0
    for key, count in counts.items():
        if count > max_count:
            max_count = count
            highest_possible_solution = key

    # Convert string to array
    X = np.fromiter(highest_possible_solution, dtype=int)
    X = X[::-1]
    #print(f'Most probable solution: {highest_possible_solution}')
    return X
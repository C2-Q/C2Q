from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector
from qiskit.primitives import Estimator
from qiskit import transpile
from qiskit_aer import AerSimulator

from scipy.optimize import minimize

import numpy as np

from qiskit.quantum_info import SparsePauliOp

def initialize_parameters(reps):
    theta = []

    # Initialize a parameter for the "gamma" and "beta" variables
    initial_param = np.pi
    
    initial_param_list = [initial_param] * reps
    theta.extend(initial_param_list)
    
    return theta

def cost_func_vqe(theta, circuit, hamiltonian, estimator = Estimator()):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    cost = estimator.run(circuit, hamiltonian, theta).result().values[0]

    return cost

def optimize_parameters(qc, ising, theta):
    # Save the expectation values the optimization gives us so that we can visualize the optimization
    exp_value_list = []

    # Here we can change the optimization method etc.
    min_minimized_optimization = minimize(cost_func_vqe, theta, method="Powell", options={'maxiter':500, 'maxfev':500}, args=(qc, ising))

    # Save the objective value the optimization finally gives us
    minimum_objective_value = min_minimized_optimization.fun
    min_exp_value_list = exp_value_list

    return min_minimized_optimization.x, minimum_objective_value

def convert_qubo_to_ising(qubo):
    # Number of qubits
    n = len(qubo)

    # Calculate the offset also, this is not important for the QAOA optimization
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
                    ising_value = -(1 / 2) * np.sum(qubo[i])
                else:
                    pauli_operator[i] = "Z"
                    pauli_operator[j] = "Z"
                    ising_value = (1 / 2) * qubo[i][j]

                if not ising_value == 0:
                    ising_pauli_op = (''.join(pauli_operator), ising_value)
                    operator_list.append(ising_pauli_op)

                offset += (1 / 2) * qubo[i][j]

    operators = SparsePauliOp.from_list(operator_list)

    return operators, offset

def vqe_no_optimization(qubo, layers):
    vqe_circuit = TwoLocal(len(qubo[0]), 'ry', 'cz', insert_barriers=True, reps=layers)
    vqe_circuit.measure_all()

    vqe_dict = {
        "qc": vqe_circuit
    }

    return vqe_dict

def vqe_optimization(qubo, layers):
    ising, offset = convert_qubo_to_ising(qubo)
    vqe_circuit = TwoLocal(len(qubo[0]), 'ry', 'cz', insert_barriers=True, reps=layers)
    num_parameters = vqe_circuit.num_parameters
    parameters = ParameterVector('theta', num_parameters)
    theta = initialize_parameters(num_parameters)
    theta, minimum_objective_value = optimize_parameters(vqe_circuit, ising, theta)
    vqe_dict = {
        "qc": vqe_circuit,
        "parameters": parameters,
        "theta": theta,
        "minimum_objective_value": minimum_objective_value
    }

    return vqe_dict

def sample_results(qc, parameters, theta, backend=AerSimulator()):
    qc_assigned_parameters = qc.assign_parameters(theta)
    qc_transpiled = transpile(qc_assigned_parameters, backend=backend)
    qc_transpiled.measure_all()

    counts = backend.run(qc_transpiled, shots=10000).result().get_counts()

    highest_possible_solution = 0
    max_count = 0
    for key, count in counts.items():
        if count > max_count:
            max_count = count
            highest_possible_solution = key

    # Convert string to array
    X = np.fromstring(highest_possible_solution, np.int8) - 48

    # Flip the bitstring to fix the order
    X = X[::-1]

    #print(f'Most probable solution: {highest_possible_solution}')
    return X
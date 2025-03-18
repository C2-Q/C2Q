from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler

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

def cost_estimator(theta, vqe_circuit_transpiled, ising, estimator, exp_value_list):
    # Calculate the expectation value

    isa_hamiltonian = ising.apply_layout(vqe_circuit_transpiled.layout)
    job = estimator.run([(vqe_circuit_transpiled, isa_hamiltonian, theta)])

    result = job.result()[0]
    cost = result.data.evs

    exp_value_list.append(cost)

    return cost

def optimize_parameters(vqe_circuit_transpiled, ising, theta, estimator):
    # Save the expectation values the optimization gives us so that we can visualize the optimization
    exp_value_list = []

    # Here we can change the optimization method etc.
    min_minimized_optimization = minimize(cost_estimator, theta, method="Powell", options={'maxiter':500, 'maxfev':500}, args=(vqe_circuit_transpiled, ising, estimator, exp_value_list))

    # Save the objective value the optimization finally gives us
    minimum_objective_value = min_minimized_optimization.fun
    min_exp_value_list = exp_value_list

    return min_minimized_optimization.x, minimum_objective_value

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

def vqe_no_optimization(qubo, layers):
    vqe_circuit = TwoLocal(len(qubo[0]), 'ry', 'cz', insert_barriers=True, reps=layers)
    vqe_circuit.measure_all()

    vqe_dict = {
        "qc": vqe_circuit
    }

    return vqe_dict

def vqe_optimization(qubo, layers, backend=AerSimulator()):
    ising, offset = convert_qubo_to_ising(qubo)
    vqe_circuit = TwoLocal(len(qubo[0]), 'ry', 'cz', insert_barriers=True, reps=layers)

    num_parameters = vqe_circuit.num_parameters
    parameters = ParameterVector('theta', num_parameters)
    theta = initialize_parameters(num_parameters)

    vqe_circuit.measure_all()
    vqe_circuit_transpiled = transpile(vqe_circuit, backend, seed_transpiler=77, layout_method='sabre', routing_method='sabre')

    estimator = Estimator(backend)
    estimator.options.default_shots = 1000

    theta, minimum_objective_value = optimize_parameters(vqe_circuit_transpiled, ising, theta, estimator)

    vqe_dict = {
        "qc": vqe_circuit,
        "parameters": parameters,
        "theta": theta,
        "minimum_objective_value": minimum_objective_value,
        "offset": offset
    }

    return vqe_dict

def sample_results(vqe_circuit, parameters, theta, backend=AerSimulator()):
    vqe_circuit_transpiled = transpile(vqe_circuit, backend, seed_transpiler=77, layout_method='sabre', routing_method='sabre')
    
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1000

    if backend.name == 'aer_simulator':
        vqe_circuit_transpiled_assigned_parameters = vqe_circuit_transpiled.decompose(reps=1).assign_parameters(theta)
    else:
        vqe_circuit_transpiled_assigned_parameters = vqe_circuit_transpiled.assign_parameters(theta)

    job = sampler.run([vqe_circuit_transpiled_assigned_parameters])

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

    #print(f'Most probable solution: {highest_possible_solution}')
    return X
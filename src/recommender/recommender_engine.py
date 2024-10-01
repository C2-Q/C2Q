# %%
from qiskit import qpy, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap

#from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.fake_provider import GenericBackendV2

import numpy as np

import os
dirname = os.path.dirname(__file__)

# %%
def load_circuit(qiskit_circ_file):
    with open(qiskit_circ_file, 'rb') as handle:
        qc = qpy.load(handle)
        
    qc = qc[0].decompose(reps=3)

    return qc

# %%
def export_circuit_data(circuit, circuit_transpiled, coupling_map):
    direct_mapping_to_device = True

    # Number of entangling gates in the original circuit
    num_entangling_gates_original = circuit.num_nonlocal_gates()

    # Number of qubits in the original circuit
    num_qubits = circuit.num_qubits

    # Number of qubits in the circuit_transpiled
    #num_qubits = circuit_transpiled.num_qubits

    # Depth of circuit_transpiled
    depth = circuit_transpiled.depth()

    # Total number of gates in the circuit_transpiled
    operations = circuit_transpiled.count_ops()
    num_operations = 0
    num_measurements = 0
    for gate, counts in operations.items():
        if not gate == "barrier" and not gate == "measure":
            num_operations += counts
        if gate == "measure":
            num_measurements += counts

    # Number of entangling gates in the circuit_transpiled
    num_entangling_gates = circuit_transpiled.num_nonlocal_gates()

    # Number of added entangling gates due to transpilation (this number can change from transpilation to transpilation since different set of qubits can be used)
    num_added_entangling_gates = num_entangling_gates - num_entangling_gates_original

    # Number of SWAP gates used in the circuit = number of added entangling gates due to transpilation divided by 3. (Three entangling gates per SWAP gate)
    num_swap_gates = int(num_added_entangling_gates/3)
    
    # If the number of SWAP gates is zero, direct mapping is possible
    if num_swap_gates > 0:
        direct_mapping_to_device = False
    
    # A list describing the qubits which have entangling gates between them. Format [c, t] where c is the control qubit and t is the target qubit
    gate_map = []

    distance_matrix = coupling_map.distance_matrix

    for gate in circuit_transpiled.data:
        if gate[0].name in ['ecr', 'cx', 'cz']:
            #print('\ngate name:', gate[0].name)
            #print('qubit(s) acted on:', gate[1])
            #print('other paramters (such as angles):', gate[0].params)

            control_gate = gate[1][0]
            target_gate = gate[1][1]
            control_qubit_index = control_gate._index
            target_qubit_index = target_gate._index

            distance = distance_matrix[control_qubit_index][target_qubit_index]
            #if distance > 1:
            #    direct_mapping_to_device = False
            #else:
            #    if abs(int(control_qubit_index) - int(target_qubit_index)) > 1:
            #        direct_mapping_to_device = False

            coupled_qubits = [int(control_qubit_index), int(target_qubit_index)]

            gate_map.append(coupled_qubits)

            #print(f'Control qubit: {control_qubit_index}')
            #print(f'Target qubit: {target_qubit_index}')

    return num_qubits, num_operations, num_measurements, num_entangling_gates, num_swap_gates, depth, gate_map, direct_mapping_to_device

# %%
device_list = [
    "IBM Eagle",
    "Rigetti Ankaa-2",
    "IQM Garnet",
    "IQM Helmi",
    "IonQ Aria",
    "Quantinuum H1",
    "Quantinuum H2",
]

# %%
provider_list = [
    "Azure Quantum",
    "IBM Quantum",
    "Amazon Braket",
    "IQM Resonance"
]

# %%
def select_provider(provider):
    if provider == "Azure Quantum":
        device_list = [
            "Rigetti Ankaa-2",
            "IonQ Aria",
            "Quantinuum H1",
            "Quantinuum H2"
        ]

        price_list_shots = [[2], [0.000220, 0.000975], [12.5], [13.5]]

    if provider == "IBM Quantum":
        device_list = [
            "IBM Eagle"
        ]

        price_list_shots = [96]
    
    if provider == "Amazon Braket":
        device_list = [
            "Rigetti Ankaa-2",
            "IonQ Aria",
            "IQM Garnet",
            "QuEra Aquila"
        ]
        
        price_list_shots = [
            0.0009,
            0.03,
            0.00145,
            0.01
        ]
    
    if provider == "IQM Resonance":
        device_list = [
            "IQM Garnet"
        ]

        price_list_shots = [30]
    
    return device_list, price_list_shots

# %%
def select_device(device):
    if device == "IBM Eagle":
        # IBM Eagle 127-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/ibm_eagle_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 127
        backend = GenericBackendV2(device_qubits, basis_gates=['ecr', 'id', 'rz', 'sx', 'x'], coupling_map=coupling_map)
        single_qubit_gate_error = 0.0002508
        two_qubit_gate_error = 0.00811
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 5.688*10**(-8)
        two_qubit_gate_timing = 5.333*10**(-7)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "Rigetti Ankaa-2":
        # Rigetti Ankaa-2 83-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/rigetti_ankaa2_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 84
        backend = GenericBackendV2(device_qubits, coupling_map=coupling_map)
        single_qubit_gate_error = 0.0015
        two_qubit_gate_error = 0.0389
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 40*10**(-9)
        two_qubit_gate_timing = 70*10**(-9)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "IQM Garnet":
        # IQM Garnet 20-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/iqm_garnet_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 20
        backend = GenericBackendV2(device_qubits, basis_gates=['id', 'r', 'cz'], coupling_map=coupling_map)
        single_qubit_gate_error = 0.0008
        two_qubit_gate_error = 0.0049
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 20*10**(-9)
        two_qubit_gate_timing = 40*10**(-9)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "IQM Helmi":
        # IQM Garnet 5-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/iqm_helmi_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 5
        backend = GenericBackendV2(device_qubits, basis_gates=['id', 'r', 'cz'], coupling_map=coupling_map)
        single_qubit_gate_error = 0.0038
        two_qubit_gate_error = 0.039
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 120*10**(-9)
        two_qubit_gate_timing = 120*10**(-9)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]
    
    if device == "IonQ Aria":
        # IonQ Aria 25-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/ionq_aria_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 25
        backend = GenericBackendV2(device_qubits, coupling_map=coupling_map)
        single_qubit_gate_error = 0.0006
        two_qubit_gate_error = 0.006
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 135*10**(-6)
        two_qubit_gate_timing = 600*10**(-6)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "Quantinuum H1":
        # Quantinuum H1 20-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/quantinuum_h1_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 20
        backend = GenericBackendV2(device_qubits, coupling_map=coupling_map)
        single_qubit_gate_error = 0.00002
        two_qubit_gate_error = 0.001
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 135*10**(-6)
        two_qubit_gate_timing = 600*10**(-6)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "Quantinuum H2":
        # Quantinuum H2 56-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/quantinuum_h2_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 56
        backend = GenericBackendV2(device_qubits, coupling_map=coupling_map)
        single_qubit_gate_error = 0.00003
        two_qubit_gate_error = 0.0015
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 135*10**(-6)
        two_qubit_gate_timing = 600*10**(-6)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]

    if device == "QuEra Aquila":
        # QuEra Aquila 256-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/quera_aquila_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 256
        backend = GenericBackendV2(device_qubits, coupling_map=coupling_map)
        single_qubit_gate_error = 0.0003
        two_qubit_gate_error = 0.005
        errors = [single_qubit_gate_error, two_qubit_gate_error]
        single_qubit_gate_timing = 135*10**(-6)
        two_qubit_gate_timing = 600*10**(-6)
        gate_timings = [single_qubit_gate_timing, two_qubit_gate_timing]
    
    return backend, coupling_map, device_qubits, device, errors, gate_timings

def recommender(qc):
    print("!!! Testing phase! The results below might be very inaccurate, do not use these to pick a quantum device !!!\n")

    num_swap_gates_list = []
    min_error = 0
    min_error_device = 0
    min_provider = 0
    min_error_price = 0

    for index_p, provider in enumerate(provider_list):
        device_list, price_list_shots = select_provider(provider)
        print(f"Provider: {provider}")
        print("")
        for index_d, device in enumerate(device_list):
            backend, coupling_map, device_qubits, device, errors, gate_timings = select_device(device)
            if device_qubits >= qc.num_qubits:
                qc_transpiled = transpile(qc, backend, seed_transpiler=100, layout_method='dense')
                num_qubits, num_operations, num_measurements, num_entangling_gates, num_swap_gates, depth, gate_map, direct_mapping_to_device = export_circuit_data(qc, qc_transpiled, coupling_map)
                
                num_swap_gates_list.append(num_swap_gates)
                
                total_average_error = (num_operations - num_entangling_gates)*errors[0] + num_entangling_gates*errors[1]

                time_to_execute = ((num_operations - num_entangling_gates)*gate_timings[0] + num_entangling_gates*gate_timings[1])*1000*50

                if provider == "Azure Quantum":
                    if device == "Rigetti Ankaa-2":
                        price_total = price_list_shots[index_d][0]*time_to_execute                
                    if device == "IonQ Aria":
                        price_iteration = ((num_operations - num_entangling_gates)*price_list_shots[index_d][0] + num_entangling_gates*price_list_shots[index_d][1])*1000
                        price_total = 50 * price_iteration
                    if device == "Quantinuum H1" or device == "Quantinuum H2":
                        price_iteration = 5+1000*(((num_operations - num_entangling_gates)+10*num_entangling_gates)/5000)
                        price_total = 50 * price_iteration * price_list_shots[index_d][0]

                if provider == "IBM Quantum":
                    price_total = price_list_shots[index_d]*(time_to_execute/60)

                if provider == "Amazon Braket":
                    price_iteration = 0.3 + price_list_shots[index_d]*1000
                    price_total = 50 * price_iteration

                if provider == "IQM Resonance":
                    price_total = price_list_shots[index_d]*(time_to_execute/60)

                if index_p == 0 and index_d == 0:
                    min_error = total_average_error
                    min_error_device = device
                    min_provider = provider
                    min_error_price = price_total
                elif total_average_error < min_error:
                    min_error = total_average_error
                    min_error_device = device
                    min_provider = provider
                    min_error_price = price_total
                    
                print(f"{device}:")

                print(f" - Single-qubit gate error: {round(errors[0]*100, 3)}%")
                print(f" - Two-qubit gate error: {round(errors[1]*100, 3)}%")
                print(f" - Calculated error when executing the circuit: {round(total_average_error*100, 2)}%")
                print(f" - Total number of gates: {num_operations}")
                print(f" - Number of entangling gates: {num_entangling_gates}")
                print(f" - SWAP gates: {num_swap_gates}")
                print(f" - Depth of the circuit: {depth}")
                print(f" - Time to execute on the quantum computer with 50 iterations and 1 000 shots: {round(time_to_execute, 6)} seconds")
                print(f" - Price with 50 iterations and 1000 shots: ${round(price_total, 2)}\n")
        
        print("--------------------------------------------\n")

    print(f"Recommended device: {min_error_device} from {min_provider} with a calculated error of {round(min_error*100, 2)}% and a price of ${round(min_error_price, 2)}.")
    print("\n!!! Testing phase! The results above might be very inaccurate, do not use these to pick a quantum device !!!")
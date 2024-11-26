from qiskit import transpile
from qiskit.transpiler import CouplingMap

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer import AerSimulator

from qiskit_ionq import IonQProvider
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

import numpy as np

from statistics import mean
import matplotlib.pyplot as plt
import math
import os
import pickle

from braket.aws import AwsDevice

import warnings

warnings.filterwarnings('ignore', message='Unable to get qubit count for ionq_simulator*')

# Obtain the working directory for importing the coupling maps
dirname = os.path.dirname(__file__)

# To get the native gate set of IonQ
ionq_provider = IonQProvider()

# Calculate number of gates.. using transpiled circuit. Return a dict of data
def export_circuit_data(circuit, circuit_transpiled):
    """
    Export numerical data from a quantum circuit which has been transpiled to a device.

    Args:
        circuit (QuantumCircuit): Original quantum circuit, without transpiling.
        circuit_transpiled (QuantumCircuit): Transpiled quantum circuit.

    Returns:
        circuit_dict (dict): Dictionary containing numerical data of the quantum circuit.
    """

    direct_mapping_to_device = True

    # Number of entangling gates in the original circuit
    num_entangling_gates_original = circuit.num_nonlocal_gates()

    # Number of qubits in the original circuit
    num_qubits = circuit.num_qubits

    # Number of qubits in the circuit_transpiled
    #num_qubits = circuit_transpiled.num_qubits

    # Total number of gates in the circuit_transpiled
    operations = circuit_transpiled.count_ops()
    num_gates = 0
    num_measurements = 0
    for gate, counts in operations.items():
        if not gate == "barrier" and not gate == "measure":
            num_gates += counts
        if gate == "measure":
            num_measurements += counts

    # Number of entangling gates in the circuit_transpiled
    num_entangling_gates = circuit_transpiled.num_nonlocal_gates()

    # Number of single-qubit gates in the circuit_transpiled
    num_single_qubit_gates = num_gates - num_entangling_gates

    # Number of added entangling gates due to transpilation (this number can change from transpilation to transpilation since different set of qubits can be used)
    num_added_entangling_gates = num_entangling_gates - num_entangling_gates_original

    # Number of SWAP gates used in the circuit = number of added entangling gates due to transpilation divided by 3. (Three entangling gates per SWAP gate)
    num_swap_gates = int(num_added_entangling_gates / 3)

    # If the number of SWAP gates is zero, direct mapping is possible
    if num_swap_gates > 0:
        direct_mapping_to_device = False

    # Total depth of circuit_transpiled
    depth = circuit_transpiled.depth()

    # Single-qubit gate layers (measurement layers included)
    single_qubit_gate_layers_meas = circuit_transpiled.depth(lambda instr: len(instr.qubits) == 1)

    # Layers containing only measurements
    measurement_layers = circuit_transpiled.depth(lambda instr: instr.operation.name == "measure")

    # Single-qubit gate layers (measurement layers not included)
    single_qubit_gate_layers = single_qubit_gate_layers_meas - measurement_layers

    # Two-qubit gate layers
    two_qubit_gate_layers = circuit_transpiled.depth(lambda instr: len(instr.qubits) == 2)

    circuit_dict = {
        "num_qubits": num_qubits,
        "gates": {
            "total": num_gates,
            "single_qubit": num_single_qubit_gates,
            "two_qubit": num_entangling_gates,
            "measurement": num_measurements,
            "swap": num_swap_gates,
        },
        "depth": {
            "total": depth,
            "single_qubit": single_qubit_gate_layers,
            "two_qubit": two_qubit_gate_layers,
            "measurement": measurement_layers
        },
        "direct_mapping": direct_mapping_to_device
    }

    return circuit_dict


# Available providers
provider_list = [
    "IBM Quantum",
    "Amazon Braket",
    "Azure Quantum",
    #"IQM Resonance"
]


# Available devices in each provider and the cost of using the devices
def select_provider(provider, available_devices):
    """
    Function which can be used to select a quantum device provider. Each provider has a specific quantum device list which are available to use. Cost to use each quantum device is defined.

    Args:
        provider (str): Quantum provider as a string

    Returns:
        device_list (list): A list containing the quantum devices available for use.
        price_list_shots (list): A list containing the cost to use each quantum device.
    """

    if provider == "Azure Quantum":
        device_list = [
            "IonQ Aria (Azure)",
            "Quantinuum H1",
            "Quantinuum H2",
            "Rigetti Ankaa-9Q-3"
        ]

        unavailable_devices = []
        for azure_device in device_list:
            if azure_device not in available_devices:
                unavailable_devices.append(azure_device)

        # https://learn.microsoft.com/en-us/azure/quantum/pricing
        price_list_shots = [[0.000220, 0.000975], [12.5], [13.5], [1.3]]

    if provider == "IBM Quantum":
        # Fetch saved devices
        device_list = []
        device_file_path = os.path.join(dirname, f'device_information/')
        loaded_ibm_saved_device_list = [filename for filename in os.listdir(device_file_path) if filename.startswith("ibm_")]
        for loaded_ibm_device in loaded_ibm_saved_device_list:
            device_list.append(loaded_ibm_device.removesuffix('.pkl'))

        # Fetch the available IBM devices
        for device in available_devices:
            if device.startswith("ibm_") and device not in device_list:
                device_list.append(device)

        price_list_shots = [96]*len(device_list) # Pay-as-you-go pricing https://www.ibm.com/quantum/pricing

    if provider == "Amazon Braket":
        device_list = [
            "IonQ Aria (Amazon)",
            "IQM Garnet",
            "Rigetti Ankaa-2"
        ]
        
        # https://aws.amazon.com/braket/pricing
        price_list_shots = [
            0.03,
            0.00145,
            0.0009
        ]

    if provider == "IQM Resonance": #not used
        device_list = [
            "IQM Garnet"
        ]

        price_list_shots = [30]
    
    if provider == "CSC": #not used yet (hard to get the calibration data, have to go through LUMI)
        device_list = [
            "IQM Helmi"
        ]

        price_list_shots = [0]

    return device_list, price_list_shots


# Define coupling map, gate errors, gate timings, T1 and T2 times for each device
def select_device(device, ibm_service, available_devices):
    """
    Function which can be used to select a quantum device. Each quantum device has its own parameters.

    Args:
        device (str): Quantum device as a string

    Returns:
        device_dict (dict): Dictionary containing numerical data of the quantum device.
    """

    if device.startswith("ibm_"):
        if ibm_service and device in available_devices:
            ibm_device = ibm_service.backend(device)
            # Fetch SX gate information
            sxlist = ibm_device.target["sx"]
            sxlist = list(sxlist.items())
            # Calculate mean SX error and duration
            sx_errorlist = []
            sx_durationlist = []
            for i in range(len(sxlist)):
                sx_errorlist.append(sxlist[i][1].error)
                sx_durationlist.append(sxlist[i][1].duration)
            single_qubit_gate_error = mean(sx_errorlist)
            single_qubit_gate_timing = mean(sx_durationlist)

            # Fetch ECR gate information
            ecrlist = ibm_device.target["ecr"]
            ecrlist = list(ecrlist.items())
            # Calculate mean ECR error and duration
            ecr_errorlist = []
            ecr_durationlist = []
            for i in range(len(ecrlist)):
                ecr_errorlist.append(ecrlist[i][1].error)
                ecr_durationlist.append(ecrlist[i][1].duration)
            two_qubit_gate_error = mean(ecr_errorlist)
            two_qubit_gate_timing = mean(ecr_durationlist)

            # Fetch measurement information
            measlist = ibm_device.target["measure"]
            measlist = list(measlist.items())
            # Calculate mean measurement error
            meas_errorlist = []
            for i in range(len(measlist)):
                meas_errorlist.append(measlist[i][1].error)
            measurement_error = mean(meas_errorlist)

            t1_list = []
            t2_list = []
            for i in range (0, 127):
                t1_list.append(ibm_device.qubit_properties(i).t1)
                t2_list.append(ibm_device.qubit_properties(i).t2)
            t1_relaxation_time = mean(t1_list)
            t2_relaxation_time = mean(t2_list)

            # IBM fake backend
            coupling_map = ibm_device.coupling_map
            device_qubits = ibm_device.num_qubits
            basis_gates = ibm_device.configuration().basis_gates
            backend = GenericBackendV2(device_qubits, basis_gates=basis_gates, coupling_map=coupling_map)

            saved_dict = {
                "name": device,
                "coupling_map": list(coupling_map),
                "basis_gates": basis_gates,
                "device_qubits": device_qubits,
                "errors": {
                    "single_qubit": single_qubit_gate_error,
                    "two_qubit": two_qubit_gate_error,
                    "measurement": measurement_error
                },
                "gate_timings": {
                    "single_qubit": single_qubit_gate_timing,
                    "two_qubit": two_qubit_gate_timing
                },
                "relaxation_times": {
                    "t1": t1_relaxation_time,
                    "t2": t2_relaxation_time
                },
            }

            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'wb') as f:
                pickle.dump(saved_dict, f)
        else:
            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'rb') as f:
                saved_dict = pickle.load(f)
            
            if available_devices:
                device = device + " (unavailable)"
            
            coupling_map = saved_dict["coupling_map"]
            basis_gates = saved_dict["basis_gates"]
            device_qubits = saved_dict["device_qubits"]
            backend = GenericBackendV2(device_qubits, basis_gates=basis_gates, coupling_map=coupling_map)
            single_qubit_gate_error = saved_dict["errors"]["single_qubit"]
            two_qubit_gate_error = saved_dict["errors"]["two_qubit"]
            measurement_error = saved_dict["errors"]["measurement"]
            single_qubit_gate_timing = saved_dict["gate_timings"]["single_qubit"]
            two_qubit_gate_timing = saved_dict["gate_timings"]["two_qubit"]
            t1_relaxation_time = saved_dict["relaxation_times"]["t1"]
            t2_relaxation_time = saved_dict["relaxation_times"]["t2"]

    if device == "Rigetti Ankaa-2":
        if available_devices and device not in available_devices:
            device = device + " (unavailable)"
        # Error rates for estimating the error
        # Calibration data from the Braket Console:
        # https://eu-north-1.console.aws.amazon.com/braket/home?region=eu-north-1#/devices/arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2
        single_qubit_gate_error = 0.001931
        two_qubit_gate_error = 0.11794
        measurement_error = 0.06565
        t1_relaxation_time = 13.012 * 10 ** (-6)
        t2_relaxation_time = 11.999 * 10 ** (-6)
        # Gate timings for estimating the time to execute a circuit
        single_qubit_gate_timing = 40*10**(-9)
        two_qubit_gate_timing = 70*10**(-9)

        # Rigetti Ankaa-2 84-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/rigetti_ankaa2_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 84
        backend = GenericBackendV2(device_qubits, basis_gates=['rx', 'rz', 'cz', 'iswap'], coupling_map=coupling_map)

    if device == "Rigetti Ankaa-9Q-3":
        if available_devices and device not in available_devices:
            device = device + " (unavailable)"
        # Error rates for estimating the error
        single_qubit_gate_error = 0.001
        two_qubit_gate_error = 0.008
        measurement_error = 0.06565 # Not found, using the Rigetti Ankaa-2 measurement error instead
        t1_relaxation_time = 21 * 10 ** (-6)
        t2_relaxation_time = 24 * 10 ** (-6)
        # Gate timings for estimating the time to execute a circuit
        single_qubit_gate_timing = 40 * 10 ** (-9) # Not found, using the Rigetti Ankaa-2 timing instead
        two_qubit_gate_timing = 70 * 10 ** (-9) # Not found, using the Rigetti Ankaa-2 timing instead

        # Rigetti Ankaa-9Q-3 9-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/rigetti_ankaa_9q_3_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 9
        backend = GenericBackendV2(device_qubits, basis_gates=['rx', 'rz', 'cz', 'iswap'], coupling_map=coupling_map)

    if device == "IQM Garnet":
        if device not in available_devices:
            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'rb') as f:
                saved_dict = pickle.load(f)
            
            if available_devices:
                device = device + " (unavailable)"

            coupling_map = saved_dict["coupling_map"]
            basis_gates = saved_dict["basis_gates"]
            device_qubits = saved_dict["device_qubits"]
            backend = GenericBackendV2(device_qubits, basis_gates=basis_gates, coupling_map=coupling_map)
            single_qubit_gate_error = saved_dict["errors"]["single_qubit"]
            two_qubit_gate_error = saved_dict["errors"]["two_qubit"]
            measurement_error = saved_dict["errors"]["measurement"]
            single_qubit_gate_timing = saved_dict["gate_timings"]["single_qubit"]
            two_qubit_gate_timing = saved_dict["gate_timings"]["two_qubit"]
            t1_relaxation_time = saved_dict["relaxation_times"]["t1"]
            t2_relaxation_time = saved_dict["relaxation_times"]["t2"]
        else:
            aws_device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")

            single_qubit_gate_error = []
            measurement_error = []
            t1_relaxation_time = []
            t2_relaxation_time = []
            
            for qubit in aws_device.properties.provider.properties["one_qubit"]:
                single_qubit_gate_error.append(1 - aws_device.properties.provider.properties["one_qubit"][qubit]["f1Q_simultaneous_RB"])
                t1_relaxation_time.append(aws_device.properties.provider.properties["one_qubit"][qubit]["T1"])
                t2_relaxation_time.append(aws_device.properties.provider.properties["one_qubit"][qubit]["T2"])
                measurement_error.append(1 - aws_device.properties.provider.properties["one_qubit"][qubit]["fRO"])
            single_qubit_gate_error = mean(single_qubit_gate_error)
            measurement_error = mean(measurement_error)
            t1_relaxation_time = mean(t1_relaxation_time)
            t2_relaxation_time = mean(t2_relaxation_time)

            two_qubit_gate_error = []
            for qubit_pair in aws_device.properties.provider.properties["two_qubit"]:
                two_qubit_gate_error.append(1 - aws_device.properties.provider.properties["two_qubit"][qubit_pair]["f2Q_simultaneous_RB_Clifford"])
            two_qubit_gate_error = mean(two_qubit_gate_error)

            # Gate timings for estimating the time to execute a circuit
            # Gate timing calibration data not available, using:
            # https://web.archive.org/web/20241126104255/https://aws.amazon.com/braket/quantum-computers/iqm/
            single_qubit_gate_timing = 20 * 10 ** (-9)
            two_qubit_gate_timing = 40 * 10 ** (-9)

            coupling_map_path = os.path.join(dirname, 'coupling_maps/iqm_garnet_coupling_map.npy')
            coupling_map = CouplingMap(np.load(coupling_map_path))
            device_qubits = 20
            basis_gates = ['id', 'r', 'cz']
            backend = GenericBackendV2(device_qubits, basis_gates=basis_gates, coupling_map=coupling_map)

            saved_dict = {
                "name": device,
                "coupling_map": list(coupling_map),
                "basis_gates": basis_gates,
                "device_qubits": device_qubits,
                "errors": {
                    "single_qubit": single_qubit_gate_error,
                    "two_qubit": two_qubit_gate_error,
                    "measurement": measurement_error
                },
                "gate_timings": {
                    "single_qubit": single_qubit_gate_timing,
                    "two_qubit": two_qubit_gate_timing
                },
                "relaxation_times": {
                    "t1": t1_relaxation_time,
                    "t2": t2_relaxation_time
                },
            }

            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'wb') as f:
                pickle.dump(saved_dict, f)

    if device == "IQM Helmi":
        # IQM Helmi 5-qubit fake backend
        coupling_map_path = os.path.join(dirname, 'coupling_maps/iqm_helmi_coupling_map.npy')
        coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 5
        backend = GenericBackendV2(device_qubits, basis_gates=['id', 'r', 'cz'], coupling_map=coupling_map)
        single_qubit_gate_error = 0.0038
        two_qubit_gate_error = 0.039
        single_qubit_gate_timing = 120 * 10 ** (-9)
        two_qubit_gate_timing = 120 * 10 ** (-9)
        t1_relaxation_time = 35.74 * 10 ** (-6)
        t2_relaxation_time = 17.464 * 10 ** (-6)

    if device == "IonQ Aria (Amazon)":
        if device not in available_devices:
            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'rb') as f:
                saved_dict = pickle.load(f)

            if available_devices:
                device = device + " (unavailable)"

            backend = ionq_provider.get_backend("simulator", gateset="native")

            single_qubit_gate_error = saved_dict["errors"]["single_qubit"]
            two_qubit_gate_error = saved_dict["errors"]["two_qubit"]
            measurement_error = saved_dict["errors"]["measurement"]
            single_qubit_gate_timing = saved_dict["gate_timings"]["single_qubit"]
            two_qubit_gate_timing = saved_dict["gate_timings"]["two_qubit"]
            t1_relaxation_time = saved_dict["relaxation_times"]["t1"]
            t2_relaxation_time = saved_dict["relaxation_times"]["t2"]
        else:
            aws_device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")

            backend = ionq_provider.get_backend("simulator", gateset="native")

            single_qubit_gate_error = 1 - aws_device.properties.provider.fidelity["1Q"]["mean"]
            two_qubit_gate_error = 1 - aws_device.properties.provider.fidelity["2Q"]["mean"]
            measurement_error = 1 - aws_device.properties.provider.fidelity["spam"]["mean"]
            single_qubit_gate_timing = aws_device.properties.provider.timing["1Q"]
            two_qubit_gate_timing = aws_device.properties.provider.timing["2Q"]
            t1_relaxation_time = aws_device.properties.provider.timing["T1"]
            t2_relaxation_time = aws_device.properties.provider.timing["T2"]

            saved_dict = {
                "name": device,
                "errors": {
                    "single_qubit": single_qubit_gate_error,
                    "two_qubit": two_qubit_gate_error,
                    "measurement": measurement_error
                },
                "gate_timings": {
                    "single_qubit": single_qubit_gate_timing,
                    "two_qubit": two_qubit_gate_timing
                },
                "relaxation_times": {
                    "t1": t1_relaxation_time,
                    "t2": t2_relaxation_time
                },
            }

            device_file_path = os.path.join(dirname, f'device_information/{device}.pkl')
            with open(device_file_path, 'wb') as f:
                pickle.dump(saved_dict, f)

        device_qubits = 25

    if device == "IonQ Aria (Azure)":
        # Using the calibration data from Amazon Braket for IonQ Aria, since the data is not available from Azure Quantum.

        device_file_path = os.path.join(dirname, f'device_information/IonQ Aria (Amazon).pkl')
        with open(device_file_path, 'rb') as f:
            saved_dict = pickle.load(f)

        if available_devices and device not in available_devices:
            device = device + " (unavailable)"

        backend = ionq_provider.get_backend("simulator", gateset="native")

        single_qubit_gate_error = saved_dict["errors"]["single_qubit"]
        two_qubit_gate_error = saved_dict["errors"]["two_qubit"]
        measurement_error = saved_dict["errors"]["measurement"]
        single_qubit_gate_timing = saved_dict["gate_timings"]["single_qubit"]
        two_qubit_gate_timing = saved_dict["gate_timings"]["two_qubit"]
        t1_relaxation_time = saved_dict["relaxation_times"]["t1"]
        t2_relaxation_time = saved_dict["relaxation_times"]["t2"]

        # IonQ Aria 25-qubit fake backend. Coupling map is not needed because the device qubits are all-to-all connected.
        #coupling_map_path = os.path.join(dirname, 'coupling_maps/ionq_aria_coupling_map.npy')
        #coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 25
        #backend = GenericBackendV2(device_qubits)

    if device == "Quantinuum H1":
        if available_devices and device not in available_devices:
            device = device + " (unavailable)"
        # Error rates for estimating the error
        # https://cdn.prod.website-files.com/669960f53cd73aedb80c8eea/6718b983d80f99dbf611b460_Quantinuum%20H1%20Product%20Data%20Sheet%20V7.00%2015Oct24.pdf
        single_qubit_gate_error = 0.00002
        two_qubit_gate_error = 0.001
        measurement_error = 0.003
        # Relaxation times
        # https://docs.quantinuum.com/h-series/support/faqs.html#machine-questions
        t1_relaxation_time = 60
        t2_relaxation_time = 4
        # Gate timings for estimating the time to execute a circuit (interzone + gate time)
        # https://arxiv.org/abs/2003.01293
        single_qubit_gate_timing = 288 * 10 ** (-6)
        two_qubit_gate_timing = 308 * 10 ** (-6)

        # Quantinuum H1 20-qubit fake backend. Coupling map is not needed because the device qubits are all-to-all connected.
        #coupling_map_path = os.path.join(dirname, 'coupling_maps/quantinuum_h1_coupling_map.npy')
        #coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 20
        #backend = GenericBackendV2(device_qubits)
        backend = QuantinuumBackend("H", machine_debug=True)

    if device == "Quantinuum H2":
        if available_devices and device not in available_devices:
            device = device + " (unavailable)"
        # Error rates for estimating the error
        # https://cdn.prod.website-files.com/669960f53cd73aedb80c8eea/6718b99685b5ef37ae2294dc_Quantinuum%20H2%20Product%20Data%20Sheet%20V2.00%2015Oct24%5B99%5D.pdf
        single_qubit_gate_error = 0.00003
        two_qubit_gate_error = 0.0015
        measurement_error = 0.0015
        # Relaxation times
        # https://docs.quantinuum.com/h-series/support/faqs.html#machine-questions
        t1_relaxation_time = 60
        t2_relaxation_time = 4
        # Gate timings for estimating the time to execute a circuit (interzone + gate time)
        # https://arxiv.org/abs/2003.01293
        single_qubit_gate_timing = 288 * 10 ** (-6)
        two_qubit_gate_timing = 308 * 10 ** (-6)

        # Quantinuum H2 56-qubit fake backend. Coupling map is not needed because the device qubits are all-to-all connected.
        #coupling_map_path = os.path.join(dirname, 'coupling_maps/quantinuum_h2_coupling_map.npy')
        #coupling_map = CouplingMap(np.load(coupling_map_path))
        device_qubits = 56
        #backend = GenericBackendV2(device_qubits)
        backend = QuantinuumBackend("H", machine_debug=True)

    device_dict = {
        "name": device,
        "backend": backend,
        "device_qubits": device_qubits,
        "errors": {
            "single_qubit": single_qubit_gate_error,
            "two_qubit": two_qubit_gate_error,
            "measurement": measurement_error
        },
        "gate_timings": {
            "single_qubit": single_qubit_gate_timing,
            "two_qubit": two_qubit_gate_timing
        },
        "relaxation_times": {
            "t1": t1_relaxation_time,
            "t2": t2_relaxation_time
        },
    }

    return device_dict

def fetch_available_devices(azure_workspace, braket_provider, ibm_service):
    # Fetch available devices
    # See tests/tests.py for an example how to use this function.

    azure_dev_list_all = azure_workspace.get_targets()

    braket_dev_list = braket_provider.backends(statuses=["ONLINE"])

    azure_dev_list = []

    for device in azure_dev_list_all:
        if "Available" in str(device):
            azure_dev_list.append(device)

    ibm_dev_list = ibm_service.backends(operational=True)

    available_devices = []

    for ibm_dev in ibm_dev_list:
        available_devices.append(ibm_dev.name)

    for braket_dev in braket_dev_list:
        if "Aria 1" in str(braket_dev):
            available_devices.append("IonQ Aria (Amazon)")
        elif "Garnet" in str(braket_dev):
            available_devices.append("IQM Garnet")

    for azure_dev in azure_dev_list:
        if "ionq.qpu.aria-1" in str(azure_dev) or "ionq.qpu.aria-2" in str(azure_dev):
            available_devices.append("IonQ Aria (Azure)")
        elif "quantinuum.qpu.h1-1" in str(azure_dev):
            available_devices.append("Quantinuum H1")
        elif "rigetti.qpu.ankaa-9q-3" in str(azure_dev):
            available_devices.append("Rigetti Ankaa-9Q-3")
    
    return available_devices

def recommender(qc, save_figures=True, ibm_service=None, available_devices=[]):
    """
    Main function of the quantum recommender. Outputs three options for quantum devices: lowest error, lowest time and lowest price.

    Args:
        qc (QuantumCircuit): The quantum circuit which is to be examined.
    Returns:
        recommender_output (string): String containing recommended quantum devices.
    """

    # Number of shots per execution and number of iterations of QAOA optimization
    num_shots = 1000
    num_iterations = 50

    #print(
    #    "!!! Testing phase! The results below might be very inaccurate, do not use these to pick a quantum device !!!\n")

    # Three metrics: minimum error, minimum time and minimum price
    min_error_device = []
    min_time_device = []
    min_price_device = []

    recommender_devices = []

    first_device = True

    for index_p, provider in enumerate(provider_list):
        device_list, price_list_shots = select_provider(provider, available_devices)
        #print(f"Provider: {provider}")
        #print("")
        for index_d, device in enumerate(device_list):
            device_dict = select_device(device, ibm_service, available_devices)
            if device_dict["device_qubits"] >= qc.num_qubits:
                # Transpile the quantum circuit to the device using a defined seed
                if "Quantinuum" in device:
                    qc_tk = qiskit_to_tk(qc)
                    compiled_qc = device_dict["backend"].get_compiled_circuit(qc_tk)
                    qc_transpiled = tk_to_qiskit(compiled_qc)
                else:
                    qc_transpiled = transpile(qc, device_dict["backend"], seed_transpiler=77, layout_method='sabre',
                                          routing_method='sabre')
                circuit_dict = export_circuit_data(qc, qc_transpiled)

                # Time to execute single shot: T = single_qubit_gate_layers * single_qubit_gate_time + two_qubit_gate_layers * two_qubit_gate_time
                time_to_execute_single_shot = circuit_dict["depth"]["single_qubit"] * device_dict["gate_timings"]["single_qubit"] + circuit_dict["depth"]["two_qubit"] * device_dict["gate_timings"]["two_qubit"]
                time_to_execute = time_to_execute_single_shot * num_shots * num_iterations

                # T1 decay
                t1_decay = 1 - (math.e ** (-time_to_execute_single_shot / device_dict["relaxation_times"]["t1"]))

                # T2 decay
                t2_decay = 1 - (math.e ** (-time_to_execute_single_shot / device_dict["relaxation_times"]["t2"]))

                # Error: E_tot = 1 - ((1 - p_1)^N_1 * (1 - p_2)^N_2 * (1 - p_m)^N_m)
                total_average_error = 1 - (((1 - device_dict["errors"]["single_qubit"]) ** circuit_dict["gates"]["single_qubit"]) * ((1 - device_dict["errors"]["two_qubit"]) ** circuit_dict["gates"]["two_qubit"]) * ((1 - device_dict["errors"]["measurement"]) ** circuit_dict["gates"]["measurement"]))
                # RB benchmarking already takes T1 and T2 relaxation into account
                #* (1 - t1_decay)**circuit_dict["num_qubits"] * (1 - t2_decay)**circuit_dict["num_qubits"])

                # Azure Quantum pricing is different on every provider
                if provider == "Azure Quantum":
                    if device.startswith("Rigetti Ankaa-9Q-3"):
                        price_total = price_list_shots[index_d][0]*time_to_execute

                    # IonQ Aria pricing is based on the amount of single-qubit and two-qubit gates
                    if device.startswith("IonQ Aria (Azure)"):
                        price_iteration = (circuit_dict["gates"]["single_qubit"] * price_list_shots[index_d][0] +
                                           circuit_dict["gates"]["two_qubit"] * price_list_shots[index_d][
                                               1]) * num_shots
                        # Azure Quantum has a defined minimum price per execution for IonQ Aria. The defined price below is for the case with error mitigation on.
                        if price_iteration < 97.5:
                            price_iteration = 97.5
                        price_total = num_iterations * price_iteration

                    # Quantinuum's pricing is based on the amount of single-qubit and two-qubit gates
                    if device.startswith("Quantinuum"):
                        price_iteration = 5 + num_shots * ((circuit_dict["gates"]["single_qubit"] + 10 *
                                                            circuit_dict["gates"]["two_qubit"]) / 5000)
                        price_total = num_iterations * price_iteration * price_list_shots[index_d][0]

                # IBM Quantum's pricing is based on the time taken on the quantum computer
                if provider == "IBM Quantum":
                    price_total = price_list_shots[index_d] * (time_to_execute / 60)

                # Amazon Braket's pricing is based on the number of shots. The price per shot is different for each quantum device
                if provider == "Amazon Braket":
                    price_iteration = 0.3 + price_list_shots[index_d] * num_shots
                    price_total = num_iterations * price_iteration

                # IQM Resonance's pricing is based on the time taken on the quantum computer
                if provider == "IQM Resonance":
                    price_total = price_list_shots[index_d] * (time_to_execute / 60)

                # If this is the first device, save its information to arrays so that we have a device that we can compare other devices to.
                if (first_device and device in available_devices) or (first_device and not available_devices):
                    min_error_device = [device, provider, total_average_error, time_to_execute, price_total]
                    min_time_device = min_error_device
                    min_price_device = min_error_device
                    first_device = False

                # If the total error of this device is lower than the error of the device saved in "min_error_device" array, update the array to this device.
                if not first_device and total_average_error < min_error_device[2] and (device in available_devices or not available_devices):
                    min_error_device = [device, provider, total_average_error, time_to_execute, price_total]

                # If the time to execute of this device is lower than the time to execute of the device saved in "min_time_device" array, update the array to this device.
                if not first_device and time_to_execute < min_time_device[3] and (device in available_devices or not available_devices):
                    min_time_device = [device, provider, total_average_error, time_to_execute, price_total]

                # If the total price of this device is lower than the total price of the device saved in "min_price_device" array, update the array to this device.
                if not first_device and price_total < min_price_device[4] and (device in available_devices or not available_devices):
                    min_price_device = [device, provider, total_average_error, time_to_execute, price_total]

                #print(f"{device}:")
                #print(f" - Single-qubit gate error: {round(device_dict["errors"]["single_qubit"]*100, 3)}%")
                #print(f" - Two-qubit gate error: {round(device_dict["errors"]["two_qubit"]*100, 3)}%")
                #print(f" - Quantum volume: {device_dict['quantum_volume']}")
                #print(f" - Calculated error when executing the circuit: {round(total_average_error * 100, 2)}%")
                #print(f" - Total number of gates: {num_operations}")
                #print(f" - Number of single-qubit gates: {circuit_dict['gates']['single_qubit']}")
                #print(f" - Number of two-qubit gates: {circuit_dict['gates']['two_qubit']}")
                #print(f" - Number of measurement gates: {circuit_dict['gates']['measurement']}")
                #print(f" - SWAP gates: {num_swap_gates}")
                #print(f" - Depth of the circuit: {depth}")
                #print(
                #    f" - Time to execute on the quantum computer with {num_iterations} iterations and {num_shots} shots: {round(time_to_execute, 6)} seconds")
                #print(f" - Coherence T1: {coherence_t1}")
                #print(f" - Coherence T2: {coherence_t2}")
                #print(f" - Price with {num_iterations} iterations and {num_shots} shots: ${round(price_total, 2)}\n")

                device_data = {
                    "name": device_dict["name"],
                    "error": total_average_error*100,
                    "time": time_to_execute,
                    "price": price_total
                }

                recommender_devices.append(device_data)

        #print("--------------------------------------------\n")

    device_names = []
    device_errors = []
    device_times = []
    device_prices = []

    if save_figures:
        for device in recommender_devices:
            device_names.append(device["name"])
            device_errors.append(device["error"])
            device_times.append(device["time"])
            device_prices.append(device["price"])

        # Plot errors
        fig = plt.figure(figsize = (22, 5))
        plt.bar(device_names, device_errors, width = 0.4)
        plt.ylabel("Error (%)")
        plt.title("Estimated total error with each quantum computer")
        plt.tight_layout()
        plt.savefig("recommender_errors_devices.png")

        # Plot times
        fig = plt.figure(figsize = (22, 5))
        plt.bar(device_names, device_times, width = 0.4)
        plt.ylabel("Time (s)")
        plt.title("Estimated total time with each quantum computer (50 iterations, 1000 shots per iteration)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("recommender_times_devices.png")

        # Plot prices
        fig = plt.figure(figsize = (22, 5))
        plt.bar(device_names, device_prices, width = 0.4)
        plt.ylabel("Price ($)")
        plt.title("Estimated price with each quantum computer (50 iterations, 1000 shots per iteration)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("recommender_prices_devices.png")

    recommender_output = f" - Lowest error: {min_error_device[0]} from {min_error_device[1]} with a calculated error of {round(min_error_device[2] * 100, 2)}%, time to execute: {round(min_error_device[3], 6)} seconds and a price of ${round(min_error_device[4], 2)}. \n - Lowest time: {min_time_device[0]} from {min_time_device[1]} with a calculated error of {round(min_time_device[2] * 100, 2)}%, time to execute: {round(min_time_device[3], 6)} seconds and a price of ${round(min_time_device[4], 2)}. \n - Lowest price: {min_price_device[0]} from {min_price_device[1]} with a calculated error of {round(min_price_device[2] * 100, 2)}%, time to execute: {round(min_price_device[3], 6)} seconds and a price of ${round(min_price_device[4], 2)}."

    return recommender_output, recommender_devices

def plot_results(recommender_data_array, qubits_array):
    """
    Plot errors, time and price of different devices with the number of qubits on the x-axis.

    Args:
        recommender_data_array (array): Data from the function recommender() in an array format. The array should be formatted so that the recommender data from the circuit with lowest amount of qubits is the first array element, then the recommender data with second lowest amount of qubits and so on.
        qubits_array (array): The number of qubits in each recommender run in an array format.
    Returns:
        Saves three files named "recommender_output_errors.png", "recommender_output_times.png" and "recommender_output_prices.png".
    """
    x = qubits_array
    y = []
    min_errorbar = []
    max_errorbar = []

    device_errors = []
    device_times = []
    device_prices = []

    device_names = []

    for recommender_data in recommender_data_array:
        device_errors_qubits = []
        device_times_qubits = []
        device_prices_qubits = []

        errors = []
        times = []
        prices = []

        for device in recommender_data:
            if device["name"] not in device_names:
                device_names.append(device["name"])
            device_errors_qubits.append([device["name"], device["error"]])
            device_times_qubits.append([device["name"], device["time"]])
            device_prices_qubits.append([device["name"], device["price"]])
            errors.append(device["error"])
            times.append(device["time"])
            prices.append(device["price"])

        min_error = min(errors)
        max_error = max(errors)
        mean_error = mean(errors)
        min_diff_error = abs(mean_error - min_error)
        max_diff_error = abs(mean_error - max_error)

        min_time = min(times)
        max_time = max(times)
        mean_time = mean(times)
        min_diff_time = abs(mean_time - min_time)
        max_diff_time = abs(mean_time - max_time)

        min_price = min(prices)
        max_price = max(prices)
        mean_price = mean(prices)
        min_diff_price = abs(mean_price - min_price)
        max_diff_price = abs(mean_price - max_price)

        y.append(mean_error)
        min_errorbar.append(min_diff_error)
        max_errorbar.append(max_diff_error)

        device_errors.append(device_errors_qubits)
        device_times.append(device_times_qubits)
        device_prices.append(device_prices_qubits)

    errors_listed_per_problem_size = []
    for device_name in device_names:
        device_qubit_list = []
        for device_list in device_errors:
            for device in device_list:
                if device[0] == device_name:
                    device_qubit_list.append(device[1])
        errors_listed_per_problem_size.append(device_qubit_list)

    times_listed_per_problem_size = []
    for device_name in device_names:
        device_qubit_list = []
        for device_list in device_times:
            for device in device_list:
                if device[0] == device_name:
                    device_qubit_list.append(device[1])
        times_listed_per_problem_size.append(device_qubit_list)

    prices_listed_per_problem_size = []
    for device_name in device_names:
        device_qubit_list = []
        for device_list in device_prices:
            for device in device_list:
                if device[0] == device_name:
                    device_qubit_list.append(device[1])
        prices_listed_per_problem_size.append(device_qubit_list)

    # Stuff for plotting average errors from all devices
    #fig = plt.figure(figsize=(15, 5))
    #yerr = [min_errorbar, max_errorbar]
    #plt.title("Estimated errors (min, avg, max) of the devices when executing the circuit")
    #plt.xlabel("Qubits")
    #plt.ylabel("Error (%)")
    #plt.xticks(x)
    #plt.errorbar(x, y, yerr=yerr, capsize=3, fmt="b--o", ecolor = "black")
    #plt.savefig("recommender_output_avg.png")

    # Plot the estimated errors for each device
    markers = ["o", "8", "p", "P", "*", "h", "H", "X", "D"]

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, device_name in enumerate(device_names):
        marker = markers[i]
        
        device_qubit_len = len(errors_listed_per_problem_size[i])
        plot_x = x[:device_qubit_len]

        ax.plot(plot_x, errors_listed_per_problem_size[i], label=device_name, marker=marker)

    plt.title("Estimated errors of the devices when executing the circuit")
    plt.xlabel("Qubits")
    plt.ylabel("Error (%)")

    plt.legend()

    plt.xticks(x)

    plt.yscale("log")
    
    plt.savefig("recommender_output_errors.png")

    # Plot the estimated times for each device
    fig, ax = plt.subplots(figsize=(15, 8))

    for i, device_name in enumerate(device_names):
        marker = markers[i]

        device_qubit_len = len(times_listed_per_problem_size[i])
        plot_x = x[:device_qubit_len]

        ax.plot(plot_x, times_listed_per_problem_size[i], label=device_name, marker=marker)

    plt.title("Estimated times of the devices when executing the circuit (50 iterations, 1000 shots per iteration)")
    plt.xlabel("Qubits")
    plt.ylabel("Time (s)")

    plt.legend()

    plt.xticks(x)

    plt.yscale("log")
    
    plt.savefig("recommender_output_times.png")

    # Plot the estimated prices for each device
    fig, ax = plt.subplots(figsize=(15, 8))

    for i, device_name in enumerate(device_names):
        marker = markers[i]

        device_qubit_len = len(prices_listed_per_problem_size[i])
        plot_x = x[:device_qubit_len]

        ax.plot(plot_x, prices_listed_per_problem_size[i], label=device_name, marker="o")

    plt.title("Estimated prices of the devices when executing the circuit (50 iterations, 1000 shots per iteration)")
    plt.xlabel("Qubits")
    plt.ylabel("Price ($)")

    plt.legend()

    plt.xticks(x)

    plt.yscale("log")

    plt.savefig("recommender_output_prices.png")
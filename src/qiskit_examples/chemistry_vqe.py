"""Example VQE for computing the ground state energy of H2."""

from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.circuit.library import TwoLocal
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper
from qiskit_nature.second_q.hamiltonians import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import FreezeCoreTransformer
from qiskit_nature.converters.second_quantization import QubitConverter

# Build electronic structure problem for H2
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735', basis='sto3g')
problem = ElectronicStructureProblem(driver, [FreezeCoreTransformer()])
second_q_ops = problem.second_q_ops()
mapper = ParityMapper()
converter = QubitConverter(mapper)
qubit_op = converter.convert(second_q_ops[0], num_particles=problem.num_particles)

# Setup VQE
backend = Aer.get_backend('aer_simulator_statevector')
ansatz = TwoLocal(problem.num_spatial_orbitals * 2, 'ry', 'cz', reps=1)
vqe = VQE(ansatz, quantum_instance=backend)
result = vqe.compute_minimum_eigenvalue(qubit_op)

print('H2 ground state energy:', result.eigenvalue.real)

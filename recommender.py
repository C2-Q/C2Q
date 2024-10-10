from src.algorithms.QAOA.QAOA import qaoa_no_optimization, qaoa_optimize, sample_results
from src.recommender.recommender_engine import recommender
from src.problems.clique import Clique
from src.graph import Graph

# Define the problem (in the future comes from the parser)
graph = Graph.random_graph(num_nodes=5)
clique_problem = Clique(graph, size=4)

# Convert the problem to a QUBO matrix
clique_qubo_instance = clique_problem.to_qubo()
Q = clique_qubo_instance.get_matrix()

# Obtain the QAOA circuit
qaoa_dict = qaoa_no_optimization(Q, 1)
qc = qaoa_dict["qc"]
# Measure all qubits
qc.measure_all()

# Run the recommender
recommender(qc)

# Run QAOA on local simulator
qaoa_dict = qaoa_optimize(Q, 1)

# Obtain the parameters of the QAOA run
qc = qaoa_dict["qc"]
parameters = qaoa_dict["parameters"]
theta = qaoa_dict["theta"]

# Sample and print out the most probable solution based on the QAOA run
sample_results(qc, parameters, theta)
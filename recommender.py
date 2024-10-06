from src.algorithms.QAOA.QAOA import qaoa_no_optimization
from src.recommender.recommender_engine import recommender
from src.problems.clique import Clique
from src.graph import Graph
# for test


# Define the problem (in the future comes from the parser)
graph = Graph.random_graph(num_nodes=5)
clique_problem = Clique(graph, size=4)

# Convert the problem to a QUBO matrix
clique_qubo_instance = clique_problem.to_qubo()
Q = clique_qubo_instance.get_matrix()

# Obtain the QAOA circuit
qc, parameters, theta = qaoa_no_optimization(Q, 1)

# Run the recommender
recommender(qc)
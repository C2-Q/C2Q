import unittest

import networkx as nx

from src.algorithms.QAOA.QAOA import qaoa_no_optimization
from src.problems.max_cut import MaxCut
from src.recommender.recommender_engine import recommender, plot_results, save_recommender_csvs


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_plot_recommender(self):
        # Plot recommender results with a range of qubits

        recommender_data_array = []
        qubits_array = []

        # Define the range of qubits
        for z in range(4, 60, 2):
            print(z)
            qubits_array.append(z)

            # Generate 3-regular graphs.
            G = nx.random_regular_graph(3, z, seed=100)
            # Turn 3-regular graphs into MaxCut QUBO formulation (can be any other problem too)
            maxcut = MaxCut(G)
            qubo = maxcut.to_qubo()
            qubo = qubo.Q

            qaoa_dict = qaoa_no_optimization(qubo, layers=1)
            qc = qaoa_dict["qc"]

            # Run the recommender and append recommender_data_array
            recommender_output, recommender_devices = recommender(qc, save_figures=True)
            recommender_data_array.append(recommender_devices)

        plot_results(recommender_data_array, qubits_array)
        save_recommender_csvs(recommender_data_array, qubits_array, outdir="ex2_intermediate.csv")

    # def test_recommender(self):


if __name__ == '__main__':
    unittest.main()

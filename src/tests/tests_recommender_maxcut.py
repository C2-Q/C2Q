import unittest
from pathlib import Path

import networkx as nx

from src.algorithms.QAOA.QAOA import qaoa_no_optimization
from src.problems.max_cut import MaxCut
from src.recommender.recommender_engine import plot_results, recommender, save_recommender_csvs


def generate_recommender_maxcut_sweep(
    min_qubits=4,
    max_qubits=58,
    step=2,
    degree=3,
    graph_seed=100,
    qaoa_layers=1,
    save_device_figures=False,
    figures_dir=None,
):
    """Generate recommender outputs for a MaxCut size sweep."""
    recommender_data_array = []
    qubits_array = []

    if min_qubits > max_qubits:
        raise ValueError("min_qubits must be <= max_qubits")
    if step <= 0:
        raise ValueError("step must be > 0")

    for z in range(min_qubits, max_qubits + 1, step):
        qubits_array.append(z)
        graph = nx.random_regular_graph(degree, z, seed=graph_seed)
        qubo = MaxCut(graph).to_qubo().Q
        qc = qaoa_no_optimization(qubo, layers=qaoa_layers)["qc"]
        run_figures_dir = None
        if save_device_figures:
            base_dir = Path(figures_dir) if figures_dir else Path(".")
            run_figures_dir = str(base_dir / f"qubits_{z}")
        _, recommender_devices = recommender(
            qc,
            save_figures=save_device_figures,
            figures_dir=run_figures_dir,
        )
        recommender_data_array.append(recommender_devices)

    return recommender_data_array, qubits_array


def run_recommender_maxcut_experiment(
    outdir="ex2_recommender.csv",
    min_qubits=4,
    max_qubits=58,
    step=2,
    degree=3,
    graph_seed=100,
    qaoa_layers=1,
    save_device_figures=False,
    figures_dir=None,
):
    """Run the full MaxCut recommender sweep and export plots plus CSVs."""
    recommender_data_array, qubits_array = generate_recommender_maxcut_sweep(
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        step=step,
        degree=degree,
        graph_seed=graph_seed,
        qaoa_layers=qaoa_layers,
        save_device_figures=save_device_figures,
        figures_dir=figures_dir,
    )
    plot_results(recommender_data_array, qubits_array, outdir=outdir)
    save_recommender_csvs(recommender_data_array, qubits_array, outdir=outdir)
    return recommender_data_array, qubits_array


class MyTestCase(unittest.TestCase):
    def test_plot_recommender_smoke(self):
        recommender_data_array, qubits_array = generate_recommender_maxcut_sweep(
            min_qubits=4,
            max_qubits=6,
            step=2,
            save_device_figures=False,
        )
        self.assertEqual(qubits_array, [4, 6])
        self.assertEqual(len(recommender_data_array), 2)
        self.assertTrue(all(isinstance(run, list) for run in recommender_data_array))


if __name__ == '__main__':
    unittest.main()

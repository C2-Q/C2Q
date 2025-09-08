from src.problems.basic_arithmetic.addition import Add
from src.problems.basic_arithmetic.multiplication import Mul
from src.problems.basic_arithmetic.subtraction import Sub
from src.problems.clique import Clique
from src.problems.factorization import Factor
from src.problems.kcolor import KColor
from src.problems.max_cut import MaxCut
from src.problems.maximal_independent_set import MIS
from src.problems.minimum_vertex_cover import MVC
from src.problems.tsp import TSP

PROBLEM_CLASS = {
    "MaxCut": MaxCut,  # Maximum Cut Problem
    "MIS": MIS,  # Maximal Independent Set
    "TSP": TSP,  # Traveling Salesman Problem
    "Clique": Clique,  # Clique Problem
    "KColor": KColor,  # K-Coloring
    "Factor": Factor,  # Factorization
    "ADD": Add,  # Addition
    "MUL": Mul,  # Multiplication
    "SUB": Sub,  # Subtraction
    "VC": MVC,  # Vertex Cover
}


class Generator:
    @staticmethod
    def generate_algorithms_circuits(problem_type, data):
        """Generate QAOA and VQE circuits for the given problem type.

        Args:
            problem_type (str): Key of the problem in ``PROBLEM_CLASS``.
            data: Input json to initialize the problem.

        Returns:
            dict: Mapping of algorithm name to generated circuit.
        """
        problem_cls = PROBLEM_CLASS.get(problem_type)
        if problem_cls is None:
            raise ValueError(f"Unknown problem type: {problem_type}")

        problem = problem_cls(data)
        qaoa_circuit = problem.qaoa()
        vqe_circuit = problem.vqe()
        return {"qaoa": qaoa_circuit, "vqe": vqe_circuit}

    @staticmethod
    def report(problem_type, data):
        """Generate a PDF report for the specified problem type."""
        problem_cls = PROBLEM_CLASS.get(problem_type)
        if problem_cls is None:
            raise ValueError(f"Unknown problem type: {problem_type}")

        problem = problem_cls(data)
        problem.report()

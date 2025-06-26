# run_from_json.py

import json
import argparse

from src.parser.parser import *
from src.generator import generator
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

PROBLEMS = {
    "MaxCut": MaxCut,
    "MIS": MIS,
    "TSP": TSP,
    "Clique": Clique,
    "KColor": KColor,
    "Factor": Factor,
    "ADD": Add,
    "MUL": Mul,
    "SUB": Sub,
    "VC": MVC
}


def load_input(path):
    with open(path, 'r') as file:
        return json.load(file)


def recognize_problem_class(type):
    if PROBLEM_POOLS[type] in GRAPH_TAGS:
        return "GRAPH"
    elif PROBLEM_POOLS[type] in ARITHMETIC_TAGS:
        return "ARITHMETIC"
    return "UNKNOWN"


def main():
    global data
    parser = argparse.ArgumentParser(description="Run C2|Q⟩ pipeline from JSON input.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    args = parser.parse_args()

    print("📥 Loading problem from JSON...")
    task = load_input(args.input)
    problem_type = task["problem_type"]
    input_data = task["data"]
    config = task.get("config", {})

    print(f"🔍 Parsing problem: {problem_type}")

    problem_class = recognize_problem_class(problem_type)
    if problem_class == "GRAPH":
        data = Graph(input_data)
    elif problem_class == "ARITHMETIC":
        data = input_data
    problem = PROBLEMS[problem_type](data)
    problem.report_latex()

    print(f"⚙️ Generating the report of quantum solutions ...")



if __name__ == "__main__":
    main()

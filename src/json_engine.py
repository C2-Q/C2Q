# run_from_json.py

import json
import argparse

from parser.parser import *
from generator.generator import *
from problems.basic_arithmetic.addition import Add
from problems.basic_arithmetic.multiplication import Mul
from problems.basic_arithmetic.subtraction import Sub
from problems.clique import Clique
from problems.factorization import Factor
from problems.kcolor import KColor
from problems.max_cut import MaxCut
from problems.maximal_independent_set import MIS
from problems.minimum_vertex_cover import MVC
from problems.tsp import TSP

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
    if type in GRAPH_TAGS:
        return "GRAPH"
    elif type in ARITHMETIC_TAGS:
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
    input_data = task["json"]
    config = task.get("config", {})

    print(f"🔍 Parsing problem: {problem_type}")
    print(f"📊 Parsed json: {input_data.items()}")
    problem_class = recognize_problem_class(problem_type)
    if problem_class == "GRAPH":
        for key, value in input_data.items():
            data = Graph(value)
    elif problem_class == "ARITHMETIC":
        data = input_data
    problem = PROBLEMS[problem_type](data.G)
    problem.report_latex()

    print(f"⚙️ Generating the report of quantum solutions ...")


if __name__ == "__main__":
    main()

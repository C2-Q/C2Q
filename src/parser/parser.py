"""
author boshuai ye
"""

import ast
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from src.graph import Graph
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

# labels = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "Factor","ADD", "MUL", "SUB", "Unknown"]
# Define problem type tags
PROBLEM_TAGS = {
    "MaxCut": 0,  # Maximum Cut Problem
    "MIS": 1,  # Maximal Independent Set
    "TSP": 2,  # Traveling Salesman Problem
    "Clique": 3,  # Clique Problem
    "KColor": 4,  # K-Coloring
    "Factor": 5,  # Factorization
    "ADD": 6,  # Addition
    "MUL": 7,  # Multiplication
    "SUB": 8,  # Subtraction
    "VC": 9,  # Vertex Cover
    "Unknown": 10
}
GRAPH_TAGS = ["MaxCut", "MIS", "TSP", "Clique", "KColor", "VC"]
ARITHMETIC_TAGS = ["ADD", "MUL", "SUB"]
ALGEBRAIC_TAGS = ["Factor"]
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
# Reverse mapping, e.g., PROBLEM_POOLS[2] = "TSP"
PROBLEM_POOLS = [k for k, v in PROBLEM_TAGS.items()]
LOCAL_MODEL_REQUIRED_FILES = ("config.json", "tokenizer_config.json")
LOCAL_MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


class Parser:
    def __init__(self, model_path: str):
        """
        Initialize the parser with the tokenizer, model, and device.

        :param model_path: str - Path to the saved model directory
        """
        model_path = self._prepare_model_path(model_path)
        use_local_only = Path(model_path).is_dir()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=use_local_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=use_local_only)
        self.model.to(self.device)

    @staticmethod
    def _prepare_model_path(model_path: str) -> str:
        model_path = str(model_path)
        candidate = Path(model_path).expanduser()

        if candidate.is_dir():
            missing = [name for name in LOCAL_MODEL_REQUIRED_FILES if not (candidate / name).is_file()]
            has_weights = any((candidate / name).is_file() for name in LOCAL_MODEL_WEIGHT_FILES)
            if missing or not has_weights:
                parts = []
                if missing:
                    parts.append("missing: " + ", ".join(missing))
                if not has_weights:
                    parts.append(
                        "missing one weight file: " + " or ".join(LOCAL_MODEL_WEIGHT_FILES)
                    )
                raise FileNotFoundError(
                    "Parser model directory is incomplete at "
                    f"{candidate.resolve()} ({'; '.join(parts)}). "
                    "Download model from "
                    "https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
                )
            return str(candidate.resolve())

        # Heuristic: treat path-like strings as local paths and fail fast with clear guidance.
        is_path_like = (
            model_path.startswith(("/", ".", "~"))
            or os.sep in model_path
            or (os.altsep is not None and os.altsep in model_path)
        )
        if is_path_like:
            raise FileNotFoundError(
                "Parser model directory does not exist at "
                f"{candidate.resolve()}. Set C2Q_MODEL_PATH or place the model in "
                "src/parser/saved_models_2025_12. Download model from "
                "https://drive.google.com/file/d/11xkJgioQkVdCGykGSLjJD1CcXu76RAIB/view?usp=drive_link"
            )

        # Otherwise allow Hugging Face model ids (e.g., roberta-base).
        return model_path

    def parse(self, classical_code: str):
        """
        Parse the classical code to determine the problem type and extract relevant json.

        - Workflow:
            - First, extract a dataset and identify if any function calls' arguments are suitable.
            - Example: For a function call like {'nx.add_edges_from': [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, 3, 'matrix']},
            the process would:
                1. Check the argument [[1, 2, 3], [4, 5, 6], [7, 8, 9]]. If this format is correct (e.g., edges or a distance matrix),
                it is used as the input.
                2. If the first argument isn't suitable, move to the next argument, 'matrix', which is likely a variable name.
                3. Retrieve the variable value from the variable set using var = variables.get(name), and check if the format matches the expected one.

            - Verification:
                - To verify if a variable's value is in the correct format:
                    - For graph problems, edge lists or distance matrices like [[...]] are considered valid.
                - If no suitable variable is found that fits the required format, raise an error.

        :param classical_code: str - The input classical code snippet.
        :return: problem_type: str, json: any - Returns the identified problem type and associated json.
        """
        try:
            # Check if the classical code is syntactically correct
            tree = ast.parse(classical_code)
        except SyntaxError as e:
            print(f"Syntax Error in the provided code: {e}")
            return "Unknown", None
        # predict labels of problem
        prediction = self._predict_classical_code(classical_code=classical_code)
        problem_class = self._recognize_problem_class(prediction)
        # ast traverse and extract json
        visitor = CodeVisitor()
        visitor.visit(tree)
        vars, calls = visitor.get_extracted_data()
        # Use extracted json for specific problem types
        if problem_class == "GRAPH":
            data = self._process_graph_data(vars, calls)
        elif problem_class == "ARITHMETIC":
            data = self._process_arithmetic_data(vars, calls)
        elif problem_class == "ALGEBRAIC":
            data = self._process_algebraic_data(vars, calls)
        else:
            data = None
        return PROBLEM_POOLS[prediction], data

    def _predict_classical_code(self, classical_code: str):
        """
        Tokenize the input code, pass it through the model, and return the predicted problem type index.

        :param classical_code: str - The input classical code snippet
        :return: int - The predicted problem type index
        """
        inputs = self.tokenizer(classical_code, return_tensors="pt", padding="max_length", truncation=True).to(
            self.device)
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)  # Get probabilities
        max_prob, prediction = torch.max(probabilities, dim=-1)
        return prediction.item()

    def _recognize_problem_class(self, prediction):
        if PROBLEM_POOLS[prediction] in GRAPH_TAGS:
            return "GRAPH"
        elif PROBLEM_POOLS[prediction] in ARITHMETIC_TAGS:
            return "ARITHMETIC"
        elif PROBLEM_POOLS[prediction] in ALGEBRAIC_TAGS:
            return "ALGEBRAIC"
        return "UNKNOWN"

    def _process_graph_data(self, variables, function_calls):
        """
        Iterate over all extracted variables and attempt to create a graph. If successful, return the graph json.
        Also, iterate through function calls to check if the arguments can be used to create a graph.
        """
        # First try variables
        for var_name, var_value in variables.items():
            try:
                graph = Graph(var_value)  # Try to create a graph
                return graph  # If successful, return the graph
            except ValueError:
                continue  # Skip and try the next variable if an exception is raised

        # Now function calls
        for args in function_calls.values():
            for arg in args:
                try:
                    graph = Graph(arg)
                    if len(graph.get_G().nodes) > 0:
                        return graph  # If 👌 return graph
                    else:
                        continue
                except ValueError:
                    continue

        # If no graph could be created, return a randomly generated graph
        # print("no data extracted, generating a random graph")
        return Graph.random_graph(num_nodes=5)

    def _process_arithmetic_data(self, variables, function_calls):
        """
        Process the function calls to extract operands for arithmetic operations.
        We are only interested in function calls that have exactly two arguments,
        both of which are either integers or variables whose values are integers.

        :param variables: dict - A dictionary of variables extracted from the code.
        :param function_calls: dict - A dictionary of function calls extracted from the code.
        :return: dict - A dictionary with the extracted operands.
        """
        # Iterate through the function calls
        for func_name, args in function_calls.items():
            resolved_args = []

            # Iterate through arguments and resolve constants or variables
            for arg in args:
                if isinstance(arg, int):  # If the argument is already an integer, it's valid 😯
                    resolved_args.append(arg)
                elif isinstance(arg, str) and arg in variables:  # Check if it's a variable in the extracted variables
                    # Try to resolve the variable's value
                    var_value = variables[arg]
                    if isinstance(var_value, int):  # Check if the variable value is an integer
                        resolved_args.append(var_value)
                    else:
                        break  # If the variable is not an integer, skip this function call！！！
                else:
                    break  # If it's neither an int nor a valid variable, skip this function call

            # Check if exactly two valid arguments have been resolved
            if len(resolved_args) == 2:
                return resolved_args

        # If no valid arithmetic function call is found, raise an error or return None or return a case, a randomized case...
        return [16, 16]
        #raise ValueError("No valid arithmetic function calls with two integer arguments found.")

    def _process_algebraic_data(self, variables, function_calls):
        """
        Resolve a single integer operand for algebraic problems (e.g., Factorization).
        Strategy:
          1) Prefer obvious integer-like variables (n, num, etc.).
          2) Inspect function call arguments: accept a single integer, or a variable
             that resolves to an integer.
          3) Fallback to a safe default (35) if nothing valid is found.

        Rules:
          - n must be >= 2.
          - If n > 512, fall back to 35.
        """

        def _validate_n(v: int) -> int:
            """Return a valid n, or None if unusable."""
            if not isinstance(v, int):
                return None
            if v < 2:
                return None
            if v > 512:  # too big, fallback rule
                return random.randint(4, 64)
            return v

        # ---- 1) Try variables: prioritize common names ----
        preferred_keys = ("n", "num", "number", "N", "value", "x")
        for key in preferred_keys:
            if key in variables and isinstance(variables[key], int):
                n = _validate_n(variables[key])
                if n is not None:
                    return n

        # Otherwise, check any integer-valued variable
        for k, v in variables.items():
            if isinstance(v, int):
                n = _validate_n(v)
                if n is not None:
                    return n

        # ---- 2) Try function calls ----
        for func_name, args in function_calls.items():
            if isinstance(args, (list, tuple)):
                # single argument case
                if len(args) == 1:
                    arg = args[0]
                    if isinstance(arg, int):
                        n = _validate_n(arg)
                        if n is not None:
                            return n
                    elif isinstance(arg, str) and arg in variables and isinstance(variables[arg], int):
                        n = _validate_n(variables[arg])
                        if n is not None:
                            return n

                # multiple arguments: take first as candidate
                if len(args) >= 1:
                    arg = args[0]
                    if isinstance(arg, int):
                        n = _validate_n(arg)
                        if n is not None:
                            return n
                    elif isinstance(arg, str) and arg in variables and isinstance(variables[arg], int):
                        n = _validate_n(variables[arg])
                        if n is not None:
                            return n

        # ---- 3) Fallback ----
        return random.randint(4, 64)


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}
        self.function_calls = {}

    def visit_Assign(self, node):
        """
        Capture assignments in the code, e.g., variables or json structures.
        """
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                self._process_tuple(target, node.value)
            if isinstance(target, ast.Name):
                value_repr = self._process_value(node.value)
                self.variables[target.id] = value_repr
        self.generic_visit(node)  # Continue traversing

    def visit_Call(self, node):
        """
        Capture function calls, including object method calls like nx.add_edges_from.
        """
        func_name = self._get_function_name(node.func)
        # Process arguments of the function call
        args = [self._process_value(arg) for arg in node.args]
        # Store the function call details
        self.function_calls[func_name] = args

        self.generic_visit(node)  # Continue traversing other nodes

    def _get_function_name(self, func):
        """
        Get the function name, whether it's a direct function call or an attribute-based method call (e.g., nx.add_edges_from).
        :param func: The AST node representing the function.
        :return: The full function name as a string.
        """
        if isinstance(func, ast.Attribute):  # Handle method calls like nx.add_edges_from
            # Get the object and method name
            obj_name = func.value.id if isinstance(func.value, ast.Name) else "unknown_obj"
            return f"{obj_name}.{func.attr}"
        elif isinstance(func, ast.Name):  # Handle simple function calls like add(a, b)
            return func.id
        return "unknown_func"

    def _process_tuple(self, target, value):
        """
        Process tuple assignments like (p, q) = (8, 8).
        :param target: The tuple on the left-hand side.
        :param value: The value being assigned (could also be a tuple).
        """
        if isinstance(value, ast.Tuple):
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name) and i < len(value.elts):
                    self.variables[elt.id] = self._process_value(value.elts[i])

    def _process_value(self, value):
        """
        Process the value being assigned to variables. This could be a constant, variable, or function call.
        :param value: The value being assigned (right-hand side of an assignment).
        :return: The processed value representation.
        """
        # If the value is a constant, return the constant's value
        if isinstance(value, ast.Constant):
            return value.value

        # If the value is a variable, return the variable's name
        elif isinstance(value, ast.Name):
            return value.id

        # If the value is a unary operation (like -8), handle it properly
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            # If it's a negative number (UnaryOp with USub), return the negative value
            # -
            if isinstance(value.operand, ast.Constant):
                return -value.operand.value

        # If the value is a function call, return the function name and arguments
        elif isinstance(value, ast.Call):
            func_name = value.func.id if isinstance(value.func, ast.Name) else "unknown_func"
            args = [self._process_value(arg) for arg in value.args]
            return f"Function Call: {func_name}({', '.join(map(str, args))})"

        # If the value is a list, process each element
        elif isinstance(value, ast.List):
            return [self._process_value(elt) for elt in value.elts]

        # If the value is a tuple, process each element
        elif isinstance(value, ast.Tuple):
            return tuple(self._process_value(elt) for elt in value.elts)

        return "Unknown Value"

    def get_extracted_data(self):
        """
        Return the extracted json from the AST traversal.
        :return: dict of extracted variables, and extracted function calls (with arguments).
        """
        return self.variables, self.function_calls

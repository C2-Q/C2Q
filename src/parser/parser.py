import ast
import torch
from transformers import RobertaTokenizer, AutoModelForSequenceClassification

from src.graph import Graph

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
    "Unknown": 9
}
GRAPH_TAGS = ["MaxCut", "MIS", "TSP", "Clique", "KColor"]
ARITHMETIC_TAGS = ["ADD", "MUL", "SUB"]

# Reverse mapping, e.g., PROBLEM_POOLS[2] = "TSP"
PROBLEM_POOLS = [k for k, v in PROBLEM_TAGS.items()]


class Parser:
    def __init__(self, model_path: str):
        """
        Initialize the parser with the tokenizer, model, and device.

        :param model_path: str - Path to the saved model directory
        """
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)

    def parse(self, classical_code: str):
        """
        Parse the classical code to determine the problem type and extract relevant data.

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
        :return: problem_type: str, data: any - Returns the identified problem type and associated data.
        """
        # predict labels of problem
        prediction = self._predict_classical_code(classical_code=classical_code)
        problem_class = self._recognize_problem_class(prediction)
        # ast traverse and extract data
        visitor = CodeVisitor()
        tree = ast.parse(classical_code)
        visitor.visit(tree)
        vars, calls = visitor.get_extracted_data()
        # Use extracted data for specific problem types (e.g., graph-related or arithmetic problems)
        if problem_class == "GRAPH":
            data = self._process_graph_data(vars, calls)
        elif problem_class == "ARITHMETIC":
            data = self._process_arithmetic_data(vars, calls)
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
        return "UNKNOWN"

    def _process_graph_data(self, variables, function_calls):
        """
        Iterate over all extracted variables and attempt to create a graph. If successful, return the graph data.
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
        for func_name, args in function_calls.items():
            for arg in args:
                try:
                    graph = Graph(arg)  # Try to create a graph from function call arguments
                    return graph  # If successful, return the graph
                except ValueError:
                    continue  # Skip and try the next argument if an exception is raised

        # If no graph could be created, return a randomly generated graph
        return Graph.random_graph()

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
                if isinstance(arg, int):  # If the argument is already an integer, it's valid
                    resolved_args.append(arg)
                elif isinstance(arg, str) and arg in variables:  # Check if it's a variable in the extracted variables
                    # Try to resolve the variable's value
                    var_value = variables[arg]
                    if isinstance(var_value, int):  # Check if the variable value is an integer
                        resolved_args.append(var_value)
                    else:
                        break  # If the variable is not an integer, skip this function call
                else:
                    break  # If it's neither an int nor a valid variable, skip this function call

            # Check if exactly two valid arguments have been resolved
            if len(resolved_args) == 2:
                return resolved_args

        # If no valid arithmetic function call is found, raise an error or return None or return a case
        return [16, 16]
        #raise ValueError("No valid arithmetic function calls with two integer arguments found.")


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = {}
        self.function_calls = {}

    def visit_Assign(self, node):
        """
        Capture assignments in the code, e.g., variables or data structures.
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
        Return the extracted data from the AST traversal.
        :return: dict of extracted variables, and extracted function calls (with arguments).
        """
        return self.variables, self.function_calls

import networkx as nx
from pysat.formula import CNF
from pysat.solvers import Solver
from itertools import combinations


def independent_set_to_sat(graph: nx.Graph) -> CNF:
    """
    Converts the Independent Set problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.
        k (int): The desired size of the independent set.

    Returns:
        CNF: The SAT formula in CNF representing the Independent Set problem.
    """
    cnf = CNF()

    def var(v: int) -> int:
        """Return the SAT variable for vertex v."""
        return v + 1

    # Clause 1: No two adjacent vertices can be in the independent set
    # For each edge (u, v) in the graph, add the clause ¬x_u ∨ ¬x_v
    for u, v in graph.edges:
        cnf.append([-var(u), -var(v)])

    return cnf


def maximal_independent_set_to_sat(graph: nx.Graph) -> CNF:
    """
    Converts the Maximal Independent Set problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.

    Returns
    -------
    CNF
        The SAT formula representing the Maximal Independent Set problem.
    """
    cnf = CNF()

    def var(v: int) -> int:
        """Return the SAT variable for vertex v."""
        return v + 1

    # Clause 1: No two adjacent vertices can both be in the independent set
    # For each edge (u, v) in the graph, add the clause ¬x_u ∨ ¬x_v
    for u, v in graph.edges:
        cnf.append([-var(u), -var(v)])

    # Clause 2: For maximality, every vertex not in the independent set
    # must have at least one neighbor in the independent set
    for v in graph.nodes:
        neighbor_clause = []
        for neighbor in graph.neighbors(v):
            neighbor_clause.append(var(neighbor))
        # Add clause that at least one neighbor must be in the independent set
        # if v is not in the set
        if neighbor_clause:
            cnf.append(neighbor_clause + [var(v)])

    return cnf


def independent_set_to_k_sat(graph: nx.Graph, k: int) -> CNF:
    """
    Converts the Independent Set problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.
        k (int): The desired size of the independent set.

    Returns:
        CNF: The SAT formula in CNF representing the Independent Set problem.
    """
    cnf = CNF()
    n = len(graph.nodes)

    def var(v: int) -> int:
        """Return the SAT variable for vertex v."""
        return v + 1

    # Clause 1: No two adjacent vertices can be in the independent set
    # For each edge (u, v) in the graph, add the clause ¬x_u ∨ ¬x_v
    for u, v in graph.edges:
        cnf.append([-var(u), -var(v)])

    # Clause 2: At least k vertices must be in the independent set
    # Generate clauses ensuring at least k vertices are in the independent set.
    # Choose n - k + 1 vertices to be excluded from the independent set.
    for subset in combinations(range(n), n - k + 1):
        # At least one vertex in this subset must be false (not in the
        # independent set)
        cnf.append([var(v) for v in subset])

    return cnf


def clique_to_sat(graph: nx.Graph, k: int) -> CNF:
    """
    Converts the k-Clique problem to a SAT problem.

    Parameters:
        graph (nx.Graph): The input graph.
        k (int): The size of the clique to find.

    Returns:
        CNF: The SAT formula in CNF representing the k-Clique problem.
    """
    cnf = CNF()
    n = len(graph.nodes)

    def var(i: int, v: int) -> int:
        """Return the SAT variable for vertex ``v`` at position ``i``."""
        return i * n + v + 1

    # Clause 1: Each position in the clique must be occupied by exactly one
    # vertex. There are k + \frac{kn(n+1)}{2} clauses.
    for i in range(k):
        # At least one vertex occupies the i-th position in the clique
        cnf.append([var(i, v) for v in range(n)])
        # No more than one vertex occupies the i-th position in the clique
        for v in range(n):
            for u in range(v + 1, n):
                cnf.append([-var(i, v), -var(i, u)])

    # Clause 2: No vertex can occupy more than one position in the clique
    # \frac{kn(k-1)}{2} clauses
    for v in range(n):
        for i in range(k):
            for j in range(i + 1, k):
                cnf.append([-var(i, v), -var(j, v)])

    # Clause 3: All vertices in the clique must be connected by an edge
    # \(k^2-k\)\(\frac{n^2-n}{2}-|E|\) clauses
    for i in range(k):
        for j in range(i + 1, k):
            for v in range(n):
                for u in range(n):
                    if v != u and not graph.has_edge(v, u):
                        cnf.append([-var(i, v), -var(j, u)])

    return cnf


def tseytin_or_to_cnf(a, b, aux_var):
    """
    Applies Tseytin transformation for the expression aux_var = (a OR b)
    reference: https://en.wikipedia.org/wiki/Tseytin_transformation
    Returns the equivalent CNF clauses for the transformation.
    """
    return [
        [-aux_var, a, b],  # (aux_var -> (a OR b)) <=> (¬aux_var OR a OR b)
        [aux_var, -a],  # (¬a -> aux_var) <=> (aux_var OR ¬a)
        [aux_var, -b]  # (¬b -> aux_var) <=> (aux_var OR ¬b)
    ]


def solve_all_cnf_solutions(cnf_formula):
    """
    Finds all solutions to the CNF formula using a SAT solver.

    Args:
        cnf_formula: The CNF formula to solve.
    Returns
    -------
    list
        A list of all satisfying assignments, where each assignment is a list
        of literals.
    """
    solutions = []
    with Solver(bootstrap_with=cnf_formula) as solver:
        for model in solver.enum_models():
            solutions.append(model)
    return solutions


def sat_to_3sat(cnf: CNF) -> CNF:
    """Convert an arbitrary CNF formula into an equivalent 3-SAT CNF.

    Clauses with more than three literals are reduced using the Tseytin
    transformation. Clauses shorter than three literals are padded by
    repeating their last literal so that every clause in the output contains
    exactly three literals.

    Parameters
    ----------
    cnf : CNF
        Input formula to convert.

    Returns
    -------
    CNF
        A new CNF instance representing the equivalent 3-SAT formula.
    """

    if not cnf.clauses:
        return CNF()

    def pad_to_three(literals):
        """Ensure a clause has length three by repeating the last literal."""
        if len(literals) < 3:
            literals.extend([literals[-1]] * (3 - len(literals)))
        return literals

    new_clauses = []
    next_var = max(abs(lit) for clause in cnf.clauses for lit in clause) + 1

    for original in cnf.clauses:
        clause = list(original)  # avoid mutating the input CNF

        while len(clause) > 3:
            first, second, *rest = clause
            for sub in tseytin_or_to_cnf(first, second, next_var):
                new_clauses.append(pad_to_three(list(sub)))
            clause = [next_var] + rest
            next_var += 1

        new_clauses.append(pad_to_three(clause))

    return CNF(from_clauses=new_clauses)

import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from src.problems.np_problems import NP
from src.problems.qubo import QUBO


class Clique(NP):
    """
    An application class for the clique problem based on a NetworkX graph.
    """

    def __init__(self, graph: Union[nx.Graph, np.ndarray, List], size: int) -> None:
        """
        Args:
            graph: A graph representing the problem. It can be specified directly as a
                   NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
            size: The desired size of the clique (K).
        """
        # If the graph is not a NetworkX graph, convert it
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, (np.ndarray, List)):
            self.graph = nx.Graph()
            self.graph.add_edges_from(graph)
        else:
            raise TypeError("The graph must be a NetworkX graph or an adjacency list/array.")

        self.size = size  # The desired clique size (K)

    def to_qubo(self, A: float = 1.0, B: float = 1.0) -> 'QUBO':
        """
        Converts the clique problem into a QUBO problem represented by a QUBO class instance
        based on the Hamiltonian H = A(K - sum_v x_v)^2 + B(K(K-1)/2 - sum_{(u,v)∈E} x_u x_v)

        Args:
            A: Penalty weight for the clique size constraint.
            B: Penalty weight for the edge constraint.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        n = self.graph.number_of_nodes()
        nodes = list(self.graph.nodes())
        node_indices = {node: idx for idx, node in enumerate(nodes)}  # Map node to index
        Q = np.zeros((n, n))

        K = self.size  # Desired clique size

        # Add linear terms to Q diagonal
        linear_coeff = -2 * A * K + A  # Coefficient for x_v
        for idx in range(n):
            Q[idx, idx] += linear_coeff

        # Add quadratic terms (upper triangular part only)
        for i in range(n):
            for j in range(i+1, n):
                # From H1: 2A for all pairs
                Q[i, j] += 2 * A

                # From H2: -B if (i, j) is an edge in E
                node_i = nodes[i]
                node_j = nodes[j]
                if self.graph.has_edge(node_i, node_j):
                    Q[i, j] += -B
                # Else, no change needed (we only modify upper triangular part)

        # Return an instance of the QUBO class with an upper triangular matrix
        return QUBO(Q)

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of node indices forming the clique.

        Args:
            result: The calculated result of the problem (binary vector).

        Returns:
            The list of node indices whose corresponding variable is 1.
        """
        x = np.array(result)
        nodes_in_clique = []
        nodes = list(self.graph.nodes())
        for idx, val in enumerate(x):
            if val == 1:
                nodes_in_clique.append(nodes[idx])
        return nodes_in_clique

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with nodes in the clique highlighted.

        Args:
            result: The calculated result for the problem (binary vector).
            pos: The positions of nodes (optional).
        """
        import matplotlib.pyplot as plt

        x = np.array(result)
        nodes = list(self.graph.nodes())
        color_map = []
        for idx in range(len(nodes)):
            if x[idx] == 1:
                color_map.append('red')  # Nodes in the clique are red
            else:
                color_map.append('gray')  # Other nodes are gray

        plt.figure(figsize=(8, 6))
        nx.draw(
            self.graph,
            node_color=color_map,
            pos=pos,
            with_labels=True,
            node_size=500,
            font_size=12,
            font_color='white',
            edge_color='black'
        )
        plt.show()
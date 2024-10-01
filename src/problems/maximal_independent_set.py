import networkx
import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt


class MIS(NP):
    """
    An application class for the maximal independent set problem based on a NetworkX graph.
    """

    def __init__(self, graph: Union[nx.Graph, np.ndarray, List]) -> None:
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

        # Store nodes and mappings
        self.nodes = list(self.graph.nodes())
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def to_qubo(self, A: float = 1.0, B: float = 1.0) -> 'QUBO':
        """
        Converts the clique problem into a QUBO problem represented by a QUBO class instance
        based on the Hamiltonian H = H_A + H_B
        Was done intuitively...
         H_A = A*sum_((u,v)\in E)(x_u*x_v)
         H_B = -B*sum_x_v
        Args:
            A: Penalty weight
            B: Penalty weight.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        n = len(self.nodes)
        Q = np.zeros((n, n))
        A = 2*B

        # Add linear terms to Q diagonal
        for idx in range(n):
            Q[idx, idx] -= B

        # Add quadratic terms (upper triangular part only)
        for i in range(n):
            for j in range(i + 1, n):
                node_i = self.nodes[i]
                node_j = self.nodes[j]
                if self.graph.has_edge(node_i, node_j):
                    Q[i, j] += A
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
        for idx, val in enumerate(x):
            if val == 1:
                node_label = self.indices_node[idx]
                nodes_in_clique.append(node_label)
        return nodes_in_clique

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with nodes in the clique highlighted.
        Args:
            result: The calculated result for the problem (binary vector).
            pos: The positions of nodes (optional).
        """
        x = np.array(result)
        # Create a mapping from node labels to their corresponding x values
        node_colors = {}
        for idx, val in enumerate(x):
            node_label = self.indices_node[idx]
            if val == 1:
                node_colors[node_label] = 'red'  # Nodes in the clique are red
            else:
                node_colors[node_label] = 'gray'  # Other nodes are gray

        # Get the nodes in the order that nx.draw will use
        graph_nodes = list(self.graph.nodes())
        color_map = [node_colors[node] for node in graph_nodes]

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
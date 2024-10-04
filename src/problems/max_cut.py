import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt


class MaxCut(NP):
    """
    An application class for the maximum cut problem based on a NetworkX graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """
        Args:
            graph: A graph representing the problem. It can be specified directly as a
                   NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
        """
        # supported Graph or List of edges
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            raise TypeError("The graph must be a NetworkX graph or an adjacency list/array.")

        # Store nodes and mappings
        self.nodes = list(self.graph.nodes())
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def to_qubo(self) -> 'QUBO':
        """
        Converts the MaxCut problem into a QUBO problem represented by a QUBO class instance.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        n = len(self.nodes)
        Q = np.zeros((n, n))
        # Construct the QUBO matrix
        for i in range(n):
            node_i = self.nodes[i]
            for j in range(i + 1, n):
                node_j = self.nodes[j]
                if self.graph.has_edge(node_i, node_j):
                    # Default weight is 1
                    weight = self.graph[node_i].get(node_j).get("weight", 1)

                    # negative weights since we are minimizing
                    Q[i, i] -= weight
                    Q[j, j] -= weight
                    Q[i, j] -= -2 * weight
        return QUBO(Q)

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of node indices forming the maximum cut problem.

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
        Draw the graph with nodes in the maximum cut highlighted and show the weights of the edges.

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
                node_colors[node_label] = 'red'  # Nodes in one set of the cut are red
            else:
                node_colors[node_label] = 'gray'  # Other nodes are gray

        # Get the nodes in the order that nx.draw will use
        graph_nodes = list(self.graph.nodes())
        color_map = [node_colors[node] for node in graph_nodes]

        # Get the positions of the nodes if not provided
        if pos is None:
            pos = nx.spring_layout(self.graph)

        plt.figure(figsize=(8, 6))

        # Draw the graph
        nx.draw(
            self.graph,
            pos=pos,
            node_color=color_map,
            with_labels=True,
            node_size=500,
            font_size=12,
            font_color='white',
            edge_color='black'
        )

        # Get edge weights for labeling
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')

        # Draw the edge labels (weights)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()

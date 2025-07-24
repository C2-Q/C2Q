import random

import networkx
import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from src.problems.np_complete import NPC
from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt
from src.graph import Graph


class KColor(NPC):
    """
    An application class for the k colors problem based on a NetworkX graph.
    reference: https://arxiv.org/pdf/1302.5843
    """

    def __init__(self, graph: nx.Graph, k: int = None) -> None:
        """
        Args:
            graph: A graph representing the problem. It can be specified directly as a
                   NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
            k: The number of colors in the problem.
        """
        # If the graph is not a NetworkX graph, convert it
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        elif isinstance(graph, Graph):
            self.graph = graph.G
        else:
            raise TypeError("The graph must be a NetworkX graph.")

        if k is None:
            k = 3
        self.k = k
        self.nodes = list(self.graph.nodes())
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def to_qubo(self, A: float = 2.0) -> 'QUBO':
        """
        Converts the k-colors problem into a QUBO problem represented by a QUBO class instance
        based on the Hamiltonian H.

        Args:
            A: Penalty weight.

        Returns:
            An instance of the QUBO class representing the problem.
        """
        N = len(self.nodes)
        k = self.k
        n_vars = N * k
        Q = np.zeros((n_vars, n_vars))

        # Linear terms: For each variable x_{v,i}, coefficient is -A
        for v in self.nodes:
            v_idx = self.node_indices[v]  # Index of node v
            for i in range(k):
                s = v_idx * k + i
                Q[s, s] += -A

        # Quadratic terms:
        # 1. For each node v, for each pair of colors i<j, coefficient is 2A
        for v in self.nodes:
            v_idx = self.node_indices[v]
            for i in range(k):
                s_i = v_idx * k + i
                for j in range(i + 1, k):
                    s_j = v_idx * k + j
                    Q[s_i, s_j] += 2 * A

        # 2. For each edge (u, v) ∈ E, for each color i, coefficient is A
        for u, v in self.graph.edges():
            u_idx = self.node_indices[u]
            v_idx = self.node_indices[v]
            for i in range(k):
                s_u_i = u_idx * k + i
                s_v_i = v_idx * k + i
                Q[s_u_i, s_v_i] += A

        return QUBO(Q)

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a list of colors assigned to the nodes in the graph.

        Args:
            result: The calculated result of the problem (binary vector).
                    Format of result like: [1,0,0, 0,1,0, 0,0,1]
                    which means the first node is colored by color 1,
                    the second node is colored by color 2, and the third node is colored by color 3.

        Returns:
            The list of colors assigned to the corresponding nodes like: [1, 2, 3], ensuring that all colors are valid.
        """
        num_nodes = len(self.nodes)
        num_colors = len(result) // num_nodes
        colors = []

        for node_idx in range(num_nodes):
            found_color = False
            for color_idx in range(num_colors):
                if result[node_idx * num_colors + color_idx] == 1:
                    colors.append(color_idx + 1)
                    found_color = True
                    break

            # Handle the case where no color is found (e.g., [0,0,0] for a node)
            if not found_color:
                random_color = random.randint(1, num_colors)  # Randomly assign a valid color
                colors.append(random_color)

        return colors

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with nodes colored based on the result.

        Args:
            result: binary vector
            pos: The positions of nodes (optional). If not provided, NetworkX will calculate a layout.
        """
        # Get the positions of the nodes if not provided
        coloring_map = self.interpret(result)
        if pos is None:
            pos = nx.spring_layout(self.graph)  # Spring layout for node positioning

        # Plot the graph
        plt.figure(figsize=(8, 6))
        nx.draw(
            self.graph,
            pos=pos,
            with_labels=True,
            node_color=coloring_map,  # Use the color map for the nodes
            node_size=500,
            font_size=12,
            font_color='white',
            cmap=plt.get_cmap('rainbow'),  # Use a colormap for the colors
            edge_color='black'
        )

        # plt.show()

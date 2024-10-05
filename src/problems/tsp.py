import numpy as np
import networkx as nx
from typing import Optional, Union, List, Dict

from src.problems.np_problems import NP
from src.problems.qubo import QUBO
import matplotlib.pyplot as plt


class TSP(NP):
    """
    An application class for the Traveling Salesman Problem (TSP) based on a NetworkX graph.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """
        Args:
            graph: A graph representing the distance between cities. It can be specified directly as a
                             NetworkX graph, or as an array or list format suitable to build a NetworkX graph.
        """
        # If the input is not a NetworkX graph, convert it
        super().__init__()
        if isinstance(graph, nx.Graph):
            self.graph = graph
        else:
            raise TypeError("The distance matrix must be a NetworkX graph or an adjacency list/array.")

        # Store nodes and mappings
        self.nodes = list(self.graph.nodes())
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.indices_node = {idx: node for idx, node in enumerate(self.nodes)}

    def to_qubo(self, A: float = 1.0, B: float = 1.0) -> 'QUBO':
        """
        Converts the TSP problem into a QUBO problem represented by a QUBO class instance.
        Reference: https://arxiv.org/pdf/1302.5843
        H = H_A + H_B
        where H_A is the Hamiltonian for Hamiltonian cycles problem
        H_B is the specific Hamiltonian for TSP

        Args:
            A: Penalty weight for enforcing the constraint that each city must be visited exactly once.
            B: Penalty weight for enforcing the constraint that the path forms a valid tour.

        Returns:
            An instance of the QUBO class representing the problem.

        Note:
            The number of variables is n^2, which grows rapidly with n (number of cities).
        """
        n = len(self.nodes)
        Q = np.zeros((n * n, n * n))  # QUBO matrix (n^2 variables)
        max_weight = self._get_max_weight()
        print(max_weight)
        A = B * max_weight + 10  # Ensure A is larger than B * max(W_uv)
        # start to construct H_B
        # variables x_v^i means node v is visited ith
        # 1. Add distance cost terms (for existing edges):
        for v in range(n):
            for u in range(n):
                # take the index like in nodes list [2,3,4,1], 0 is node 2
                index_i = self.nodes[v]
                index_j = self.nodes[u]
                if index_i != index_j and self.graph.has_edge(index_i, index_j):  # Ensure (i, j) is a valid edge
                    weight = self.graph[index_i][index_j].get('weight', 1)  # Get the edge weight or default to 1
                    for p in range(n - 1):
                        Q[v * n + p, u * n + (p + 1)] += B * weight

                        # start to construct H_A
        # 2. Add penalty terms to ensure each position in the tour is assigned exactly one city
        for v in range(n):
            for j in range(n):
                #v = self.nodes[v]
                Q[v * n + j, v * n + j] -= A
                for k in range(j + 1, n):
                    Q[v * n + j, v * n + k] += 2 * A

        # 3. Add penalty terms to ensure each city is visited exactly once
        for j in range(n):
            for v in range(n):
                #v = self.nodes[v]
                Q[v * n + j, v * n + j] -= A
                for u in range(v + 1, n):
                    #u = self.nodes[u]
                    Q[v * n + j, u * n + j] += 2 * A

        # 4. last part of H_A
        for v in range(n):
            for u in range(n):
                # take the index like in nodes list [2,3,4,1], 0 is node 2
                index_i = self.nodes[v]
                index_j = self.nodes[u]
                if index_i != index_j and not self.graph.has_edge(index_i, index_j):  # Ensure edge (i, j) doesnt exist
                    for p in range(n - 1):
                        Q[v * n + p, u * n + (p + 1)] += A  # Add distance cost
        return QUBO(Q)

    def _get_max_weight(self):
        max_weight = -float('inf')
        for u in self.graph.adj:
            for v in self.graph.adj[u]:
                weight = self.graph.adj[u][v].get('weight', 1)
                if weight > max_weight:
                    max_weight = weight
        return max_weight

    def interpret(self, result: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Interpret a result as a sequence of node indices forming the optimal TSP route.

        Args:
            result: The calculated result of the problem (binary vector).

        Returns:
            The sequence of node indices representing the optimal tour.
        """
        x = np.array(result).reshape((len(self.nodes), len(self.nodes)))
        tour = []
        for t in range(len(self.nodes)):
            city = np.argmax(x[:, t])
            tour.append(self.indices_node[city])
        return tour

    def draw_result(self, result: Union[np.ndarray, List[int]], pos: Optional[Dict[int, np.ndarray]] = None) -> None:
        """
        Draw the graph with the optimal TSP tour highlighted using arrows to depict the route and showing edge weights.

        Args:
            result: The calculated result for the problem (binary vector).
            pos: The positions of nodes (optional). If not provided, NetworkX will calculate a layout.
        """
        x = np.array(result).reshape((len(self.nodes), len(self.nodes)))
        tour_edges = []
        for t in range(len(self.nodes) - 1):
            city1 = np.argmax(x[:, t])
            city2 = np.argmax(x[:, t + 1])
            tour_edges.append((self.indices_node[city1], self.indices_node[city2]))
        city1 = np.argmax(x[:, len(self.nodes) - 1])
        city2 = np.argmax(x[:, 0])
        tour_edges.append((self.indices_node[city1], self.indices_node[city2]))

        # Get the positions of the nodes if not provided
        if pos is None:
            pos = nx.spring_layout(self.graph)

        plt.figure(figsize=(8, 6))

        # Draw the graph
        nx.draw(
            self.graph,
            pos=pos,
            with_labels=True,
            node_color='lightblue',
            node_size=500,
            font_size=12,
            font_color='white',
            edge_color='black'
        )

        # Draw the TSP tour with arrows
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=tour_edges,
            edge_color='red',
            width=2,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.2'  # Adds curvature to the arrows
        )

        # Draw edge weights (for all edges)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()

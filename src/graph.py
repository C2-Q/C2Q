import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, input_data):
        """
        Initializes a Graph object that can take either a distance matrix or a list of edges.
        The input type is determined automatically based on the structure of input_data.
        If an edge has no weight, it will default to a weight of 1.

        :param input_data: Either a distance matrix (2D array) or a list of edges [(node1, node2, weight), ...]
        """
        self.input_data = input_data
        self.graph = nx.Graph()  # Create a networkx graph object

        # Determine whether input_data is a distance matrix or a list of edges
        if self._is_matrix(input_data):
            self._build_graph_from_matrix()
        elif self._is_edge_list(input_data):
            self._build_graph_from_edges()
        else:
            raise ValueError("Input data must be either a distance matrix (2D array) or a list of edges.")

    @staticmethod
    def random_graph(num_nodes=5, edge_prob=0.5, weighted=True, max_weight=10):
        """
        Generate a random graph either by using a random distance matrix or random edges.

        :param num_nodes: Number of nodes in the graph.
        :param edge_prob: Probability of creating an edge between two nodes.
        :param weighted: Whether to assign random weights to the edges.
        :param max_weight: Maximum weight of the edges if weighted.
        :return: A Graph object with random edges or a random distance matrix.
        """
        # Create random edges with probabilities
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_prob:
                    weight = np.random.randint(1, max_weight) if weighted else 1
                    edges.append((i, j, weight) if weighted else (i, j))

        if len(edges) == 0:
            raise ValueError("No edges were generated for the graph. Try increasing edge_prob.")

        return Graph(input_data=edges)

    def _is_matrix(self, data):
        """
        Determine whether the input data is a distance matrix (2D array).
        :param data: The input data.
        :return: True if the data is a matrix, False otherwise.
        """
        return isinstance(data, (list, np.ndarray)) and all(isinstance(row, (list, np.ndarray)) for row in data)

    def _is_edge_list(self, data):
        """
        Determine whether the input data is a list of edges.
        Each edge should be a tuple (node1, node2, weight) or (node1, node2) with default weight 1.
        :param data: The input data.
        :return: True if the data is a valid list of edges, False otherwise.
        """
        return isinstance(data, list) and all(
            isinstance(edge, tuple) and (len(edge) == 2 or len(edge) == 3) for edge in data)

    def _build_graph_from_matrix(self):
        """
        Build the graph from a distance matrix.
        """
        matrix = np.array(self.input_data)  # Convert input_data to a NumPy array
        num_nodes = len(matrix)

        # Add edges with weights from the distance matrix
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Use i + 1 to avoid duplicating edges
                if matrix[i][j] > 0:  # Only add edges where the weight is greater than 0
                    self.graph.add_edge(i, j, weight=matrix[i][j])

    def _build_graph_from_edges(self):
        """
        Build the graph from a list of edges.
        The edges can be in the form [(node1, node2)] or [(node1, node2, weight)].
        If no weight is provided, a default weight of 1 will be assigned.
        """
        for edge in self.input_data:
            if len(edge) == 2:  # If only (node1, node2) is provided, assign weight 1
                self.graph.add_edge(edge[0], edge[1], weight=1)
            elif len(edge) == 3:  # If (node1, node2, weight) is provided, use the given weight
                self.graph.add_edge(edge[0], edge[1], weight=edge[2])

    def visualize(self):
        """
        Visualize the graph using matplotlib.
        """
        pos = nx.spring_layout(self.graph)  # Layout for node positioning
        nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=15)

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)

        plt.show()

    def get_graph(self):
        """
        Return the networkx graph object.
        """
        return self.graph



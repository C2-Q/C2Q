import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, input_data):
        """
        Initializes a Graph object that can take either a distance matrix
        or a list of edges.

        - Matrix: square 2D array-like of shape (n, n)
        - Edge list: list of (u, v) or (u, v, weight), where each edge
          can be a tuple or list.

        If an edge has no weight, it will default to a weight of 1.
        """
        self.input_data = input_data
        self.G = nx.Graph()  # Create a networkx graph object

        # Reject completely empty input early
        if input_data is None:
            raise ValueError("Input json must not be None.")
        if isinstance(input_data, (list, np.ndarray)) and len(input_data) == 0:
            raise ValueError("Input json must not be an empty list/matrix.")

        # Decide representation type
        if self._is_matrix(input_data):
            self._build_graph_from_matrix()
        elif self._is_edge_list(input_data):
            self._build_graph_from_edges()
        else:
            raise ValueError(
                "Input json must be either a square distance/adjacency matrix "
                "(2D array) or a list of edges."
            )

    @staticmethod
    def random_graph(num_nodes=3, edge_prob=0.5, weighted=True, max_weight=10):
        """Generate and return a ``Graph`` instance with random edges."""
        import numpy as np
        import networkx as nx

        num_nodes = max(1, num_nodes)

        tmp_graph = nx.Graph()
        tmp_graph.add_nodes_from(range(num_nodes))

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < edge_prob:
                    weight = np.random.randint(1, max_weight) if weighted else 1
                    tmp_graph.add_edge(i, j, weight=weight)

        if tmp_graph.number_of_edges() == 0 and num_nodes > 1:
            u, v = np.random.choice(num_nodes, 2, replace=False)
            weight = np.random.randint(1, max_weight) if weighted else 1
            tmp_graph.add_edge(u, v, weight=weight)

        edges_with_weights = [
            (u, v, data.get("weight", 1))
            for u, v, data in tmp_graph.edges(data=True)
        ]

        return Graph(edges_with_weights)

    def _is_matrix(self, data):
        """
        Determine whether the input json is an adjacency/distance matrix.

        We require:
        - list or np.ndarray
        - non-empty
        - every row is list/np.ndarray
        - square: len(row) == number of rows, for all rows
        """
        if not isinstance(data, (list, np.ndarray)):
            return False
        if len(data) == 0:
            return False

        # Convert to array to inspect shape robustly
        arr = np.array(data, dtype=object)

        # If it's clearly not 2D, it's not a matrix
        if arr.ndim != 2:
            return False

        n_rows, n_cols = arr.shape
        if n_rows != n_cols:
            return False

        # Also ensure each row is list-like (for safety with weird objects)
        for row in data:
            if not isinstance(row, (list, np.ndarray)):
                return False

        return True

    def _is_edge_list(self, data):
        """
        Determine whether the input json is a list of edges.

        Accepts:
        - list of tuples or lists
        - each item has length 2 or 3:
            (u, v) or (u, v, weight)
        """
        if not isinstance(data, list):
            return False
        if len(data) == 0:
            return False

        for edge in data:
            if not isinstance(edge, (list, tuple)):
                return False
            if len(edge) not in (2, 3):
                return False
        return True

    def _build_graph_from_matrix(self):
        """
        Build the graph from a distance/adjacency matrix.
        Matrix is assumed to be square (n x n) where entry (i, j) is the weight.
        """
        matrix = np.array(self.input_data)
        num_nodes = matrix.shape[0]

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # avoid duplicate edges
                w = matrix[i, j]
                try:
                    # treat None / non-numeric as "no edge"
                    if w is None:
                        continue
                except Exception:
                    pass
                if w > 0:
                    self.G.add_edge(i, j, weight=float(w))

    def _build_graph_from_edges(self):
        """
        Build the graph from a list of edges.
        Edges can be:
            [(u, v), (u, v, w), ...]
        with u, v node ids and w the weight.
        If no weight is provided, a default weight of 1 is used.
        """
        for edge in self.input_data:
            if len(edge) == 2:
                u, v = edge
                self.G.add_edge(u, v, weight=1)
            elif len(edge) == 3:
                u, v, w = edge
                self.G.add_edge(u, v, weight=w)

    def visualize(self):
        """Visualize the graph using matplotlib."""
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=15)

        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        plt.show()

    def get_G(self):
        """Return the networkx graph object."""
        return self.G
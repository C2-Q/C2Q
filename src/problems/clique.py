from src.problems.np_problems import *


class Clique(NP):
    def __init__(self, graph: Graph, size=3):
        super().__init__()
        self.three_sat = None
        self.graph = graph
        self.sat = clique_to_sat(graph.graph, size)
        self.size = size


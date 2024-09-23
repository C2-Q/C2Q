from src.problems.np_problems import *


class IS(NP):
    def __init__(self, graph: Graph, size=3):
        super().__init__()
        self.graph = graph
        self.sat = independent_set_to_sat(graph.graph, size)
        self.size = size




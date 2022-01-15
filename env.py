import networkx as nx
import numpy as np

def make_env(args):
    env = Environment()
    return env, args

class Environment:
    def __init__(self,):
        self.graph = nx.Graph()
        edges = [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6)]
        self.graph.add_nodes_from(range(7))
        self.graph.add_edges_from(edges)
        # print(graph.edges)

    def generate_traffic_matrix(self):
        traffic_matrix = np.random.normal(5,1,[7,7]).astype(np.int)
        # print(traffic_matrix)
        return traffic_matrix

    def reset(self):
        pass

    def step(self):
        pass


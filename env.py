import networkx as nx
import numpy as np
import copy as cp

from data_processor import DataProcesser


# def make_env(args):
#     env = Environment()
#     return env, args


class Environment:
    def __init__(self, num_paths):
        self.graph = nx.Graph()
        edges = [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6)]
        self.graph.add_nodes_from(range(7))
        self.graph.add_edges_from(edges)
        self.num_nodes = nx.number_of_nodes(self.graph)
        self.num_edges = nx.number_of_edges(self.graph)
        self.dp = DataProcesser(self.graph)
        self.dp.all_k_shortest_paths(num_paths)
        self.last_tm = np.zeros([self.num_nodes, self.num_nodes])
        self.counter = 0
        self.num_paths = num_paths
        self.reset()

        # print(graph.edges)

    def generate_traffic_matrix(self):
        traffic_matrix = np.random.normal(5, 1, [7, 7]).astype(np.int16)
        # print(traffic_matrix)
        return traffic_matrix

    def reset(self):
        self.bandwidth = np.ones(self.num_edges, dtype=np.int16) * 100
        self.rest_bd = np.ones(self.num_edges, dtype=np.int16) * 100
        self.counter = 0
        new_tm = self.generate_tm()
        self.last_tm = new_tm
        obs = DataProcesser.actor_inputs(self.rest_bd,new_tm)
        return obs

    def step(self, actions):
        self.counter += 1
        rest_bd = cp.deepcopy(self.bandwidth)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                occupy = DataProcesser.split_to_traffic(i, j, actions[(i, j)], self.last_tm[i, j])
                rest_bd -= occupy
                # print(rest_bd)
        utility = 1 - rest_bd / self.bandwidth
        # print(self.bandwidth)
        # print(utility)
        reward = -np.round(np.max(utility), decimals=3)
        if self.counter >= 20:
            done = True
        else:
            done = False
        new_tm = self.generate_tm()
        self.last_tm = new_tm
        # print(new_tm)
        obs = DataProcesser.actor_inputs(rest_bd,new_tm)
        return obs, reward, done

    def generate_tm(self):
        tm = np.random.uniform(5, 8, size=[7, 7]).astype(np.int16)
        for i in range(self.num_nodes):
            tm[i,i] = 0
        return tm

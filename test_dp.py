from data_processor import DataProcesser
import networkx as nx
import numpy as np

graph = nx.Graph()
edges = [(0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5), (3, 6), (4, 6), (5, 6)]
graph.add_nodes_from(range(7))
graph.add_edges_from(edges)

dp = DataProcesser(graph)
# print(dp._link_to_node)
dp.all_k_shortest_paths(3)

paths,idx, seq = DataProcesser.get_paths_inputs(0,2)
# print(paths,idx, seq)

tm = np.random.uniform(5,15,size=[7,7]).astype(np.int16)
print(tm)
bw = [20,20,20,20,20,20,20,20,20,20,20,20]
feature = DataProcesser.actor_inputs(bw,tm)
print(feature.shape)
print(feature)

actions = np.ones([1,126])*0.3
state = feature*actions
state = np.nansum(state,axis=1)
print(state)
print(state.shape)
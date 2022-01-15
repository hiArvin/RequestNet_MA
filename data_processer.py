import networkx as nx
import numpy as np

class DataProcesser:
    # 数据写成静态变量

    def __init__(self,node_graph):
        self.node_graph = node_graph
        self.edge_graph = self.node_graph_to_edge_graph(node_graph)

    def node_graph_to_edge_graph(self,node_graph):
        edge_graph = None
        return edge_graph

    def all_k_shortest_paths(self):
        pass

    def k_shortest_paths(self,src,dst):
        pass

    def get_paths(self,src,dst):
        pass

    def selection_to_path(self,src,dst,selection):
        pass

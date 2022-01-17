import networkx as nx
import copy as cp
import numpy as np

class DataProcesser:
    _sd_pair_to_paths={}

    def __init__(self,node_graph):
        self.node_graph = node_graph
        self.edge_graph = self.node_graph_to_edge_graph(node_graph)

    def node_graph_to_edge_graph(self, node_graph):
        edge_graph = None
        return edge_graph

    def all_k_shortest_paths(self):
        for node_i in self.node_graph:
            for node_j in self.node_graph:
                if node_i==node_j:
                    continue
                paths = self.k_shortest_paths(node_i,node_j,k=3)
                self._sd_pair_to_paths[(node_i, node_j)]=paths


    def k_shortest_paths(self, source, target, k=1, weight='weight'):
        # G is a networkx graph.
        # source and target are the labels for the source and target of the path.
        # k is the amount of desired paths.
        # weight = 'weight' assumes a weighed graph. If this is undesired, use weight = None.

        A = [nx.dijkstra_path(self.node_graph, source, target, weight='weight')]
        # A_len = [sum([self.node_graph[A[0][l]][A[0][l + 1]]['weight'] for l in range(len(A[0]) - 1)])]
        B = []

        for i in range(1, k):
            for j in range(0, len(A[-1]) - 1):
                Gcopy = cp.deepcopy(self.node_graph)
                spurnode = A[-1][j]
                rootpath = A[-1][:j + 1]
                for path in A:
                    if rootpath == path[0:j + 1]:  # and len(path) > j?
                        if Gcopy.has_edge(path[j], path[j + 1]):
                            Gcopy.remove_edge(path[j], path[j + 1])
                        if Gcopy.has_edge(path[j + 1], path[j]):
                            Gcopy.remove_edge(path[j + 1], path[j])
                for n in rootpath:
                    if n != spurnode:
                        Gcopy.remove_node(n)
                try:
                    spurpath = nx.dijkstra_path(Gcopy, spurnode, target, weight='weight')
                    totalpath = rootpath + spurpath[1:]
                    if totalpath not in B:
                        B += [totalpath]
                except nx.NetworkXNoPath:
                    continue
            if len(B) == 0:
                break
            lenB = [len(path) for path in B]
            B = [p for _, p in sorted(zip(lenB, B))]
            A.append(B[0])
            # A_len.append(sorted(lenB)[0])
            B.remove(B[0])

        return A

    @classmethod
    def get_paths_inputs(cls, src,dst):
        paths = cls._sd_pair_to_paths.get((src, dst))
        pt=[]
        idx = []
        seq = []
        for i,p in enumerate(paths):
            for j, n in enumerate(p):
                pt.append(n)
                idx.append(i)
                seq.append(j)
        return pt, idx, seq

    @classmethod
    def selection_to_path(cls,src,dst,selection):
        paths = cls._sd_pair_to_paths.get((src, dst))
        path = paths[selection]
        return path


import networkx as nx
import copy as cp
import numpy as np


class DataProcesser:
    _sd_pair_to_paths = {}
    _link_to_node = {}
    _paths_mask = {}  # dict, (src,dst):np array with shape, [num_nodes,num_paths]

    def __init__(self, node_graph):
        self.node_graph = node_graph
        self.edge_graph = self.node_graph_to_edge_graph(node_graph)

        for idx, edge in enumerate(nx.edges(self.node_graph)):
            src,dst = edge
            self._link_to_node[edge] = idx
            self._link_to_node[(dst,src)] = idx

    def node_graph_to_edge_graph(self, node_graph):
        edge_graph = None
        return edge_graph

    def all_k_shortest_paths(self, num_paths):
        n_nodes = nx.number_of_nodes(self.node_graph)
        for node_i in range(n_nodes):
            for node_j in range(n_nodes):
                if node_i == node_j:
                    continue
                paths = self.k_shortest_paths(node_i, node_j, k=num_paths)
                DataProcesser._sd_pair_to_paths[(node_i, node_j)] = paths
                paths_np = self.path_to_array(paths)
                DataProcesser._paths_mask[(node_i,node_j)] = paths_np

    def k_shortest_paths(self, source, target, k=1, weight='weight'):
        A = [nx.dijkstra_path(self.node_graph, source, target, weight='weight')]
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

    def path_to_array(self, paths):
        path_array = np.zeros([nx.number_of_edges(self.node_graph), len(paths)], dtype=np.int16)
        for j, p in enumerate(paths):
            for i in range(len(p) - 1):
                idx = self._link_to_node[(p[i], p[i + 1])]
                path_array[idx][j] = 1
        return path_array

    @classmethod
    def get_paths_inputs(cls, src:int, dst:int):
        paths = cls._sd_pair_to_paths[(src, dst)]
        trans_paths = []
        for p in paths:
            trans_p = []
            for i in range(len(p) - 1):
                trans_p.append(cls._link_to_node[p[i], p[i + 1]])
            trans_paths.append(trans_p)
        pt = []
        idx = []
        seq = []
        for i, p in enumerate(trans_paths):
            for j, n in enumerate(p):
                pt.append(n)
                idx.append(i)
                seq.append(j)
        return pt, idx, seq

    @classmethod
    def split_to_traffic(cls, src, dst, split, traffic):
        num_paths = len(split)
        traffic_size = split * traffic
        mask = cp.deepcopy(cls._paths_mask[(src,dst)])
        for i in range(num_paths):
            mask[:, i] = mask[:, i] * traffic_size[i]
        occupy = np.sum(mask,axis=1)
        return occupy

    @classmethod
    def actor_inputs(cls, bw, tm):
        '''
        tm 矩阵
        :param bw:
        :param tm:
        :return:
        '''
        paths_np = cp.deepcopy(cls._paths_mask)
        n_sd = len(paths_np)
        n_edges,n_paths = paths_np[(0,1)].shape
        for sd_pair,p_mask in paths_np.items():
            src,dst = sd_pair
            paths_np[sd_pair] = p_mask*tm[src][dst]

        request_np = np.concatenate(list(paths_np.values()),axis=1)

        # 防0作为分母
        bw = np.array(bw)
        bw[bw == 0] = 1
        bw = np.expand_dims(bw, -1)
        input_features = request_np / np.tile(bw, [1, n_sd * n_paths])
        return input_features

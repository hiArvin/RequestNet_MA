import tensorflow as tf
from test_layer import PathEmbedding, FlowPointer


class Actor:
    def __init__(self, src, dst, num_paths, paths, idx, seq, theta1, theta2, theta3):
        self.src = src
        self.dst = dst
        self.num_path = num_paths
        self.pem_layer = PathEmbedding(num_paths=num_paths,
                                       path_state_dim=theta1,
                                       paths=paths,
                                       index=idx,
                                       sequences=seq)
        self.ptr_layer = FlowPointer(hidden_dim1=theta2, hidden_dim2=theta3)

    def forward(self, input):
        path_embedding = self.pem_layer(input)
        outs = self.ptr_layer(path_embedding)
        return outs


class Critic:
    def __init__(self):
        pass

    def forward(self,states, actions):
        pass


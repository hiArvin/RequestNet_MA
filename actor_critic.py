import tensorflow as tf
# from tensorflow import Module
from layers import PathEmbedding, FlowPointer


class Actor(tf.Module):
    def __init__(self, src, dst, num_paths, paths, idx, seq, theta1, theta2, theta3,name=None):
        super(Actor, self).__init__(name=name)

        self.src = src
        self.dst = dst
        self.num_path = num_paths
        self.layers= []
        with self.name_scope:
            self.layers.append(PathEmbedding(num_paths=num_paths,
                                       path_state_dim=theta1,
                                       paths=paths,
                                       index=idx,
                                       sequences=seq))
            self.layers.append(FlowPointer(hidden_dim1=theta2, hidden_dim2=theta3))

    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Critic(tf.Module):
    def __init__(self):
        super(Critic, self).__init__()
        pass

    def forward(self,states, actions):
        '''
        这里的actions输入之后需要做数据处理，不然结果不会好了
        '''
        return 0


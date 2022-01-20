import tensorflow as tf
from tensorflow.keras.layers import Dense
# from tensorflow import Module
from layers import PathEmbedding, FlowPointer


class Actor(tf.Module):
    '''
        单个预测版本，要改成batch版本
    '''

    def __init__(self, num_paths, paths, idx, seq, theta1=3, theta2=2, name=None):
        super(Actor, self).__init__(name=name)
        self.num_path = num_paths
        self.layers = []
        with self.name_scope:
            self.layers.append(PathEmbedding(num_paths=num_paths,
                                             path_state_dim=theta1,
                                             paths=paths,
                                             index=idx,
                                             sequences=seq))
            self.layers.append(FlowPointer(hidden_dim1=theta2))

    @tf.Module.with_name_scope
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Critic(tf.Module):
    def __init__(self,hidden=3):
        super(Critic, self).__init__()
        self.layers = []
        with self.name_scope:
            self.layers.append(Dense(hidden))
            self.layers.append(Dense(1))

    @tf.Module.with_name_scope
    def __call__(self, states):
        '''
        这里的actions输入之后需要做数据处理，不然结果不会好了
        '''
        for layer in self.layers:
            states = layer(states)
        x = tf.reduce_sum(states,axis=-1)
        return x

import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D
# from tensorflow import Module
from layers import PathEmbedding, FlowPointer, PEM


class Actor(tf.Module):
    '''
        单个预测版本，要改成batch版本
    '''

    def __init__(self, num_paths, paths, idx, seq, theta1=3, theta2=2, name=None):
        super(Actor, self).__init__(name=name)
        self.num_path = num_paths
        self.layers = []
        with self.name_scope:
            self.layers.append(PEM(num_paths=num_paths,
                                   path_state_dim=theta1,
                                   paths=paths,
                                   index=idx,
                                   sequences=seq))
            self.layers.append(FlowPointer(hidden_dim1=theta2))

    @tf.Module.with_name_scope
    def __call__(self, x):
        x = tf.cast(x, dtype=tf.float32)
        batch, _, _ = tf.shape(x)
        pem_out = []
        for b in range(batch):
            pem_out.append(self.layers[0](x[b, :]))
        out = self.layers[1](pem_out)
        return out

class ActorSimple(tf.Module):
    def __init__(self, num_edges,total_paths,num_paths,name=None):
        super(ActorSimple, self).__init__(name=name)
        self.layer1 = Conv2D(filters=3,kernel_size=(3,3),input_shape=(num_edges,total_paths),activation="relu",trainable=True)
        self.layer2 = Dense(64,trainable=True)
        self.layer3 = Dense(num_paths,trainable=True)

    @tf.Module.with_name_scope
    def __call__(self, x):
        shape = tf.shape(x)
        batch_size = shape[0]
        x= tf.expand_dims(x, axis=-1)
        x1 = self.layer1(x)
        x1 = tf.reshape(x1, [batch_size,-1])
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = tf.nn.softmax(x3)
        return x4


class Critic(tf.Module):
    def __init__(self, hidden=3):
        super(Critic, self).__init__()
        self.layers = []
        with self.name_scope:
            self.layers.append(Dense(hidden,trainable=True))
            self.layers.append(Dense(1,trainable=True))

    @tf.Module.with_name_scope
    def __call__(self, states):
        '''
        这里的actions输入之后需要做数据处理，不然结果不会好了
        '''
        for layer in self.layers:
            states = layer(states)
        x = tf.reduce_sum(states, axis=-1)
        return x

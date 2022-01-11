import tensorflow as tf
from tensorflow.keras.layers import Layer


class PathEmbedding(Layer):
    def __init__(self, num_paths, path_state_dim, paths, index, sequences,
                 act=None, **kwargs):
        super(PathEmbedding, self).__init__(**kwargs)
        self.num_paths = num_paths
        self.paths = paths
        self.idx = index
        self.seqs = sequences
        self.path_state_dim = path_state_dim
        self.act = act

    def build(self, input_shape):
        self.num_edges, self.link_state_dim = input_shape
        initializer = tf.keras.initializers.GlorotUniform()
        # gru cell
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.path_update.build(tf.TensorShape([None, self.link_state_dim]))
        self.rnn_layer = tf.keras.layers.RNN(self.path_update, return_sequences=True, return_state=True)

        # attention
        self.wq = initializer([self.path_state_dim, self.path_state_dim])
        self.wk = initializer([self.path_state_dim, self.path_state_dim])
        self.wv = initializer([self.path_state_dim, self.path_state_dim])

    def call(self, inputs):
        # RNN
        h_tild = tf.gather(inputs, self.paths)
        ids = tf.stack([self.idx, self.seqs], axis=1)
        max_len = tf.reduce_max(self.seqs) + 1
        shape = tf.stack([self.num_paths, max_len, self.link_state_dim])
        lens = tf.math.segment_sum(data=tf.ones_like(self.idx),
                                   segment_ids=self.idx)
        link_inputs = tf.scatter_nd(ids, h_tild, shape)

        hidden_states, last_state = self.rnn_layer(link_inputs)

        key = tf.matmul(hidden_states, self.wk)
        query = tf.matmul(last_state, self.wq)
        value = tf.matmul(hidden_states, self.wv)
        self.att = tf.matmul(key, tf.expand_dims(query, -1))
        self.att = tf.transpose(self.att, [0, 2, 1])
        context = tf.matmul(self.att, value)
        return context


class FlowPointer(Layer):
    def __init__(self,hidden_dim1, hidden_dim2,**kwargs):
        super(FlowPointer, self).__init__(**kwargs)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

    def build(self, input_shape):
        print(input_shape)
        self.num_paths, _, self.path_state_dim = input_shape
        initializer = tf.keras.initializers.GlorotUniform()
        # Trainable parameters

        self.RNN = tf.keras.layers.SimpleRNN(self.hidden_dim1, return_sequences=True, return_state=True)
        self.wq = initializer([self.hidden_dim1, self.hidden_dim2])
        self.wk = initializer([self.hidden_dim1, self.hidden_dim2])


    def call(self, inputs):
        # path_state = tf.reshape(inputs, [1, self.num_paths, self.path_state_dim])
        hidden_state, flow_state = self.RNN(inputs)
        key = tf.matmul(hidden_state, self.wk)
        query = tf.matmul(flow_state, self.wq)
        att = tf.matmul(key, tf.expand_dims(query, -1))
        att = tf.squeeze(att)

        # att = tf.nn.softmax(att)
        return att

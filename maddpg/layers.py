import tensorflow as tf
import copy as cp
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
        self.batch_size, self.num_edges, self.link_state_dim = input_shape
        # gru cell
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.path_update.build(tf.TensorShape([None, self.link_state_dim]))
        self.rnn_layer = tf.keras.layers.RNN(self.path_update, return_sequences=True, return_state=True)

        # attention
        self.wq = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_q')
        self.wk = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_k')
        self.wv = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_v')

    def call(self, inputs):
        # RNN
        h_tild = tf.gather(inputs, self.paths, axis=1)
        batch = tf.range(self.batch_size)
        batch = tf.tile(tf.expand_dims(batch, axis=1), (1, len(self.idx)))
        batch = tf.expand_dims(batch, -1)
        ids = tf.stack([self.idx, self.seqs], axis=1)
        ids = tf.tile(tf.expand_dims(ids, 0), (self.batch_size, 1, 1))
        ids = tf.concat([batch, ids], axis=-1)
        max_len = tf.reduce_max(self.seqs) + 1
        shape = tf.stack([self.batch_size, self.num_paths, max_len, self.link_state_dim])
        link_inputs = tf.scatter_nd(ids, h_tild, shape)
        link_inputs = tf.reshape(link_inputs, [self.batch_size * self.num_paths, max_len, self.link_state_dim])
        hidden_states, last_state = self.rnn_layer(link_inputs)

        key = tf.matmul(hidden_states, self.wk)
        query = tf.matmul(last_state, self.wq)
        value = tf.matmul(hidden_states, self.wv)
        self.att = tf.matmul(key, tf.expand_dims(query, -1))
        self.att = tf.transpose(self.att, [0, 2, 1])
        context = tf.matmul(self.att, value)
        context = tf.reshape(context, [self.batch_size, self.num_paths, self.path_state_dim])
        return context


class PEM(Layer):
    def __init__(self, num_paths, path_state_dim, paths, index, sequences,
                 act=None, **kwargs):
        super(PEM, self).__init__(**kwargs)
        self.num_paths = num_paths
        self.paths = paths
        self.idx = index
        self.seqs = sequences
        self.path_state_dim = path_state_dim
        self.act = act

    def build(self, input_shape):
        self.num_edges, self.link_state_dim = input_shape
        # gru cell
        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.path_update.build(tf.TensorShape([None, self.link_state_dim]))
        self.rnn_layer = tf.keras.layers.RNN(self.path_update, return_sequences=True, return_state=True)
        # attention
        self.wq = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_q')
        self.wk = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_k')
        self.wv = self.add_weight(shape=[self.path_state_dim, self.path_state_dim],
                                  initializer=tf.keras.initializers.GlorotUniform,
                                  trainable=True, name='att_v')

    def call(self, inputs):
        total_paths = max(self.idx)
        print(self.idx)
        print(total_paths)
        # RNN
        h_tild = tf.gather(inputs, self.paths)
        print(tf.shape(h_tild))
        ids = tf.stack([self.idx, self.seqs], axis=1)
        print(ids)
        max_len = tf.reduce_max(self.seqs) + 1
        shape = tf.stack([total_paths, max_len, self.link_state_dim])
        print(shape)
        lens = tf.math.segment_sum(data=tf.ones_like(self.idx),
                                   segment_ids=self.idx)
        link_inputs = tf.scatter_nd(ids, h_tild, shape)

        hidden_states, last_state = self.rnn_layer(self.path_update,
                                                   link_inputs,
                                                   sequence_length=lens,
                                                   dtype=tf.float32)


        key = tf.matmul(hidden_states, self.wk)
        query = tf.matmul(last_state, self.wq)
        value = tf.matmul(hidden_states, self.wv)
        self.att = tf.matmul(key, tf.expand_dims(query, -1))
        self.att = tf.transpose(self.att, [0, 2, 1])
        context = tf.matmul(self.att, value)
        return context


class FlowPointer(Layer):
    def __init__(self, hidden_dim1, hidden_dim2=1, **kwargs):
        super(FlowPointer, self).__init__(**kwargs)
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

    def build(self, input_shape):
        self.num_paths, _, self.path_state_dim = input_shape
        initializer = tf.keras.initializers.GlorotUniform()
        # Trainable parameters

        self.RNN = tf.keras.layers.SimpleRNN(self.hidden_dim1, return_sequences=True, return_state=True)
        self.wq = initializer([self.hidden_dim1, self.hidden_dim2])
        self.wk = initializer([self.hidden_dim1, self.hidden_dim2])

    def call(self, inputs):
        hidden_state, flow_state = self.RNN(inputs)
        key = tf.matmul(hidden_state, self.wk)
        query = tf.matmul(flow_state, self.wq)
        att = tf.matmul(key, tf.expand_dims(query, -1))
        att = tf.squeeze(att)

        att = tf.nn.softmax(att)
        return att

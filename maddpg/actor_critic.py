import tensorflow as tf
from tensorflow.keras.layers import Dense,Layer

class Critic(tf.keras.Model):
    def __init__(self,num_agents, total_paths, max_len, link_state_dim, path_state_dim=7):
        super(Critic, self).__init__()
        self.num_agents = num_agents
        self.num_paths = total_paths
        self.max_len = max_len
        self.link_state_dim = link_state_dim
        self.path_state_dim = path_state_dim
        self.pem = PEM(path_state_dim)
        self.fc1 = Dense(32,activation='sigmoid')
        self.fc2 = Dense(32,activation='sigmoid')
        self.fc3 = Dense(16,activation='sigmoid')
        self.out = Dense(1)

    def call(self, inputs):
        states, pointers, actions = inputs
        pointers = tf.reshape(pointers, [-1, self.num_agents*self.num_agents*2])
        actions = tf.reshape(actions, [-1, self.num_paths])

        states = tf.concat(states,axis=1)
        path_state = self.pem(states)
        path_state = tf.reshape(path_state, [-1, self.num_paths*self.path_state_dim])

        c_state = tf.concat([path_state,pointers,actions],axis=-1)
        c_state = self.fc1(c_state)
        c_state = self.fc2(c_state)
        c_state = self.fc3(c_state)
        c_state = self.out(c_state)
        return c_state


class RequestNet(tf.keras.Model):
    def __init__(self, num_paths, path_state_dim, name='RequestNet'):
        super(RequestNet, self).__init__(name=name)
        self.pem = PEM(path_state_dim)
        self.fc1 = Dense(path_state_dim, activation="relu")
        self.flow_ptr_layer = FlowPointer(num_paths, path_state_dim, path_state_dim)

    def call(self, inputs):
        # 输入的时候要注意修改形式
        x, pointer = inputs
        path_features = self.pem(x)
        # print("path_features:\n", path_features.numpy())
        pointer_ft = self.fc1(pointer)
        output = self.flow_ptr_layer([path_features, pointer_ft])
        return output


class PEM(Layer):
    def __init__(self, path_state_dim,
                 act=None, **kwargs):
        super(PEM, self).__init__(**kwargs)
        self.path_state_dim = path_state_dim
        # self.link_state_dim = link_state_dim
        self.act = act

    def build(self, input_shape):
        self.batch_size, self.num_paths, self.max_len, self.link_state_dim = input_shape
        # gru cell
        # self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        # self.path_update.build(tf.TensorShape([None, self.link_state_dim]))
        self.rnn_layer = tf.keras.layers.SimpleRNN(self.path_state_dim, return_sequences=True, return_state=True,activation='sigmoid')
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
        inputs = tf.reshape(inputs, [-1, self.max_len, self.link_state_dim])
        # mask = np.tile(np.expand_dims(np.array([1,3,1,3]),axis=0),[self.batch_size,1])
        hidden_states, last_state = self.rnn_layer(inputs)
        key = tf.sigmoid(tf.matmul(hidden_states, self.wk))
        query = tf.sigmoid(tf.matmul(last_state, self.wq))
        value = tf.sigmoid(tf.matmul(hidden_states, self.wv))
        self.att = tf.matmul(key, tf.expand_dims(query, -1))
        self.att = tf.transpose(self.att, [0, 2, 1])
        context = tf.matmul(self.att, value)
        # print(self.num_paths)
        output = tf.reshape(context, [-1, self.num_paths, self.path_state_dim])
        return output


class FlowPointer(Layer):
    def __init__(self, num_path, path_state_dim, hidden_dim, **kwargs):
        super(FlowPointer, self).__init__(**kwargs)
        self.num_path = num_path
        self.path_state_dim = path_state_dim
        self.hidden_dim = hidden_dim
        initializer = tf.keras.initializers.GlorotUniform()
        # Trainable parameters

        self.wq = initializer([self.path_state_dim, self.hidden_dim])
        self.wk = initializer([self.path_state_dim, self.hidden_dim])

    def call(self, inputs):
        hidden_state, flow_state = inputs
        key = tf.sigmoid(tf.matmul(hidden_state, self.wk))
        query = tf.sigmoid(tf.matmul(flow_state, self.wq))
        query = tf.expand_dims(query, -1)
        att = tf.einsum("bij,bjk->bik", key, query)
        att = tf.squeeze(att)
        att = tf.reshape(att, [-1, self.num_path])

        att = tf.nn.softmax(att)
        # att = tf.reshape(att,[-1,self.num_sd_pair*self.num_path])
        return att

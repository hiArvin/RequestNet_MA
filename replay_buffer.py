import threading
import numpy as np


class Buffer:
    def __init__(self, args, num_nodes, obs_shape, action_shape):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        self.num_nodes =num_nodes
        self.buffer['o'] = np.empty([self.size, obs_shape[0],obs_shape[1]])
        self.buffer['o_next'] = np.empty([self.size, obs_shape[0],obs_shape[1]])
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                # self.buffer['o_%d_%d' % (i, j)] = np.empty([self.size, obs_shape])
                self.buffer['u_%d_%d' % (i, j)] = np.empty([self.size, action_shape])
                self.buffer['r_%d_%d' % (i, j)] = np.empty([self.size])
                # self.buffer['o_next_%d_%d' % (i, j)] = np.empty([self.size, obs_shape])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        self.buffer['o'][idxs] = o
        self.buffer['o_next'][idxs] = o_next
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i==j: continue
                with self.lock:
                    # self.buffer['o_%d_%d' % (i, j)][idxs] = o[i]
                    self.buffer['u_%d_%d' % (i, j)][idxs] = u[(i,j)]
                    self.buffer['r_%d_%d' % (i, j)][idxs] = r[(i,j)]
                    # self.buffer['o_next_%d_%d' % (i, j)][idxs] = o_next[i]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx

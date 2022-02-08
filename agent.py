import numpy as np
import os
from maddpg import MADDPG


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Agent:
    def __init__(self, args, src, dst):
        self.args = args
        self.policy = MADDPG(args, src, dst)

    def select_action(self, o, epsilon):
        # TODO: noise rate
        if np.random.uniform() < epsilon:
            u = np.random.uniform(0,1, [self.args.num_paths])
            u = softmax(u)
            # print('random action')
        else:
            # inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(o)
            u = pi
            # print('agent action')
            # print('{} : {}'.format(self.name, pi))
            # u = pi.cpu().numpy()
            # pi
            # noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            # u += noise
            # u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)


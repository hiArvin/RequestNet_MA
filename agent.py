import numpy as np
import os
from maddpg.maddpg import MADDPG
# from common.replay_buffer import ReplayBuffer


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Agent:
    def __init__(self,agent_id, args):
        self.agent_id = agent_id
        self.args = args
        self.policy = MADDPG(agent_id, args)
        # self.buffer = ReplayBuffer(args)

    def select_action(self, o, epsilon):
        # TODO: noise rate
        if np.random.uniform() < epsilon:
            u = np.random.uniform(0., 1., self.args.num_paths[self.agent_id])
            u = u / np.sum(u)
        else:
            # inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network.predict(o).squeeze()
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


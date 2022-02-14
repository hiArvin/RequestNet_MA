import tqdm
import numpy as np

from replay_buffer import Buffer
from agent import Agent

class Runner:
    def __init__(self,args,env):
        self.args = args
        self.num_paths = args.num_paths
        self.env = env
        self.num_nodes = env.num_nodes
        self.num_edges = env.num_edges
        self.episode_limit = 20
        self.epsilon = 0.1

        self.agents = self._init_agents()
        self.buffer = Buffer(args,
                             self.num_nodes,
                             [self.num_edges,self.num_paths*self.num_nodes*(self.num_nodes-1)],
                             self.num_paths)


    def _init_agents(self):
        agents = {}
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i==j:
                    continue
                agent = Agent(self.args,i,j)
                agents[(i,j)]=agent
        return agents

    def run(self):
        obs = self.env.reset()
        for time_step in range(5):
            # reset the environment
            if time_step % self.episode_limit == 0:
                obs = self.env.reset()
            u = []
            actions = {}
            for sd_pair, agent in self.agents.items():
                action = agent.select_action(np.expand_dims(obs,axis=0),self.epsilon)
                actions[sd_pair]=action
            obs_next,rewards,done=self.env.step(actions)
            self.buffer.store_episode(obs,actions,rewards,obs_next)
            obs = obs_next
            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for idx, agent in self.agents.items():
                    other_agents = self.agents.copy()
                    del other_agents[idx]
                    agent.learn(transitions, other_agents)







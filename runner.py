import numpy as np
from tqdm import tqdm
from common.replay_buffer import ReplayBuffer
from agent import Agent
import matplotlib.pyplot as plt

class Runner:
    def __init__(self,args,env):
        self.args = args
        self.num_paths = args.num_paths
        self.env = env
        self.episode_limit = 20
        self.epsilon = 0.1
        self.save_path = self.args.save_dir
        self.agents = self._init_agents()
        self.replay_buffer = self._init_replay_buffer()
        self.num_experiences = 0

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def _init_replay_buffer(self):
        replaybuffer = []
        for i in range(self.args.n_agents):
            rb = ReplayBuffer(i, self.args.buffer_size)
            replaybuffer.append(rb)
        return replaybuffer

    def run(self):
        returns = []
        obs, pt = self.env.reset()
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                obs, pt = self.env.reset()
            u = []
            actions = []
            for agent_id, agent in enumerate(self.agents):
                input_state = [np.expand_dims(obs[agent_id],axis=0), np.expand_dims(pt[agent_id],axis=0)]
                action = agent.select_action(input_state,self.epsilon)
                u.append(action)
                actions.append(action)
            rewards, obs_pt1, pt_tp1, done, info = self.env.step(actions)
            for agent_id in range(self.args.n_agents):
                self.replay_buffer[agent_id].add(
                    obs[agent_id],
                    pt[agent_id],
                    actions[agent_id],
                    rewards[agent_id],
                    obs_pt1[agent_id],
                    pt_tp1[agent_id],
                    done
                )
            self.num_experiences += 1
            # self.buffer.add(obs,pt,actions,rewards,obs_pt1,pt_tp1,done)
            obs = obs_pt1
            pt = pt_tp1
            if self.num_experiences >= self.args.batch_size:
                idxes = [np.random.randint(0, self.num_experiences - 1) for _ in range(self.args.batch_size)]
                transitions = []
                for agent_id in range(self.args.n_agents):
                    transitions.append(self.replay_buffer[agent_id].encode_sample(idxes))
                for id, agent in enumerate(self.agents):
                    other_agents = self.agents.copy()
                    del other_agents[id]
                    agent.learn(transitions, other_agents)

            if time_step > 0 and time_step % self.args.evaluate_rate == 0:
                returns.append(self.evaluate())

            # self.noise = max(0.05, self.noise - 0.0005)
            self.epsilon = max(0.05, self.epsilon - 0.0005)
            # np.save(self.save_path + '/returns.pkl', returns)
        plt.figure()
        plt.plot(range(len(returns)), returns)
        plt.xlabel('episode * ' + str(self.args.evaluate_rate / self.args.time_steps))
        plt.ylabel('average returns')
        plt.savefig(self.save_path + '/'+self.args.env_name +'.png', format='png')

    def evaluate(self):
        returns = []
        for episode in range(self.args.evaluate_episodes):
            # reset the environment
            obs, pt = self.env.reset_eval()
            rewards = 0
            done = False
            action_all=[]
            while not done:
                # self.env.render()
                actions = []
                for agent_id, agent in enumerate(self.agents):
                    input_state = [np.expand_dims(obs[agent_id], axis=0), np.expand_dims(pt[agent_id], axis=0)]
                    action = agent.select_action(input_state,0)
                    actions.append(action)
                # print(np.array(actions))
                # for i in range(self.args.n_agents):
                #     actions.append([0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0])
                r, obs_pt1, pt_tp1, done, info = self.env.step_eval(actions)
                rewards += r[0]
                obs = obs_pt1
                pt = pt_tp1
            # print("Actions:",np.average(action_all,axis=0))
            returns.append(rewards)
        avg_rew = sum(returns) / self.args.evaluate_episodes
        print("Average Reward is: ",avg_rew)
        return avg_rew




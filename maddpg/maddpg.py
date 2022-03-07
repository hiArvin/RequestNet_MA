import numpy as np
import tensorflow as tf
import os
from maddpg.actor_critic import RequestNet, Critic


class MADDPG:
    def __init__(self, agent_id, args):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # create the network
        self.actor_network = RequestNet(num_paths=args.num_paths[agent_id],
                                        path_state_dim=args.path_state_dim[agent_id])
        self.critic_network = Critic(num_agents=args.n_agents, total_paths=sum(args.num_paths), path_state_dim=3,
                                     link_state_dim=7, max_len=args.max_len)

        # build up the target network
        self.actor_target_network = RequestNet(num_paths=args.num_paths[agent_id],
                                               path_state_dim=args.path_state_dim[agent_id])
        self.critic_target_network = Critic(num_agents=args.n_agents, total_paths=sum(args.num_paths), path_state_dim=3,
                                            link_state_dim=7, max_len=args.max_len)

        # # load the weights into the target networks
        # self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        # self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # create the optimizer
        # initialize optimizers
        # lr_actor, lr_critic = 0.001, 0.0001
        self.actor_optim = tf.compat.v1.train.AdamOptimizer(self.args.lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(self.args.lr_critic)
        self.critic_network.compile(self.critic_optim, loss='mse')

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        # self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        # if not os.path.exists(self.model_path):
        #     os.mkdir(self.model_path)
        # self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        # if not os.path.exists(self.model_path):
        #     os.mkdir(self.model_path)

        # # 加载模型
        # if os.path.exists(self.model_path + '/actor_params.pkl'):
        #     self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
        #     self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
        #     print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
        #                                                                   self.model_path + '/actor_params.pkl'))
        #     print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
        #                                                                    self.model_path + '/critic_params.pkl'))

    # soft update
    def _soft_update_target_network(self):
        for target_variable, variable in zip(self.actor_target_network.variables, self.actor_network.variables):
            target_variable.assign((1 - self.args.tau) * target_variable + self.args.tau * variable)
        for target_variable, variable in zip(self.critic_target_network.variables, self.critic_network.variables):
            target_variable.assign((1 - self.args.tau) * target_variable + self.args.tau * variable)

    # update the network
    def train(self, transitions, other_agents):
        # transitions: 0:obses_t 1:pts_t  2:actions  3:rewards 4:obses_tp1 5:pts_tp1 6:dones
        obs, pt, actions, r, obs_pt1, pt_tp1, done = transitions[self.agent_id]
        obs_all, pt_all, actions_all, obs_pt1_all, pt_tp1_all = [], [], [], [], []
        for i in range(self.args.n_agents):
            obs_all.append(transitions[i][0])
            pt_all.append(transitions[i][1])
            actions_all.append(transitions[i][2])
            obs_pt1_all.append(transitions[i][4])
            pt_tp1_all.append(transitions[i][5])
        # calculate the target Q value function
        action_next = []

        # 得到下一个状态对应的动作
        # print(other_agents)
        index = 0
        for agent_id in range(self.args.n_agents):
            if agent_id == self.agent_id:
                action_next.append(self.actor_target_network.predict([obs_pt1, pt_tp1]))
            else:
                # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                actor_inputs = [obs_pt1_all[agent_id], pt_tp1_all[agent_id]]
                action_next.append(other_agents[index].policy.actor_target_network(actor_inputs))
                index += 1
        action_next = np.concatenate([action_next[i] for i in range(self.args.n_agents)], axis=-1)
        obs_pt1_inputs = np.concatenate(obs_pt1_all,axis=1)
        pt_tp1_all_inputs = np.concatenate(pt_tp1_all,axis=1)
        q_next = self.critic_target_network.predict([obs_pt1_inputs, pt_tp1_all_inputs, action_next])
        # print("reward:",r )
        target_q = r + self.args.gamma * q_next
        # print(target_q)
        obs_all_inputs = np.concatenate(obs_all, axis=1)
        pt_all_inputs = np.concatenate(pt_all, axis=1)
        actions_all_inputs = np.concatenate(actions_all, axis=1,dtype=np.float32)
        # print(pt_all_inputs.shape)
        hist = self.critic_network.fit([obs_all_inputs, pt_all_inputs, actions_all_inputs], target_q, epochs=1, verbose=0)
        # print("c_loss:",hist.history["loss"])
        critic_loss = hist.history['loss'][0]

        a_pt_all_input = np.array(pt_all).transpose((1,0,2))
        # a_actions_all_input = np.array(actions_all).transpose((1,0,2))
        actor_loss = self._update_actor_network(obs_all_inputs, a_pt_all_input, actions_all_inputs)
        # print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    @tf.function
    def _update_actor_network(self, obs, pt, actions):
        cord1 = sum(self.args.num_paths[:self.agent_id])
        cord2 = cord1+self.args.num_paths[self.agent_id]
        with tf.GradientTape() as tape:
            # the actor loss
            # 重新选择联合动作中当前agent的动作，其他agent的动作不变
            x = self.actor_network([obs[:,cord1:cord2], pt[:,self.agent_id]])
            # actions = tf.unstack(actions)
            if self.agent_id == 0:
                new_actions = tf.concat([x, actions[:, cord2:]], axis=1)
            elif self.agent_id == self.args.n_agents:
                new_actions = tf.concat([actions[:, :cord1], x], axis=1)
            else:
                new_actions = tf.concat([actions[:, :cord1], x, actions[:,cord2:]], axis=1)
            actor_loss = - tf.reduce_mean(self.critic_network([obs, pt, new_actions]))
            # if self.agent_id == 0:
            #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
            # update the network
            # back propagation ac net
        actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        # print(actor_loss)
        # print(actor_grad)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor_network.trainable_weights))
        # critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
        # self.critic_optim.apply_gradients(zip(critic_grad, self.critic_network.trainable_variables))
        return actor_loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # tf.Save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        # tf.save(self.critic_network.state_dict(),  model_path + '/' + num + '_critic_params.pkl')

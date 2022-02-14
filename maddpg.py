import tensorflow as tf
import numpy as np
from data_processor import DataProcesser
from actor_critic import Actor, Critic, ActorSimple


class MADDPG:
    def __init__(self, args, src, dst):
        self.args = args
        self.src = src
        self.dst = dst
        self.num_paths = args.num_paths
        self.num_edges = args.num_edges
        self.num_nodes = args.num_nodes

        self.tau = 1.0 - 1e-2

        paths, idxs, seqs = DataProcesser.get_paths_inputs(src, dst)
        # create network
        # self.actor_network = Actor(args.num_paths, paths, idxs, seqs)
        num_request = self.num_nodes*(self.num_nodes-1)
        self.actor_network = ActorSimple(self.num_edges,self.num_paths*num_request,self.num_paths)

        self.critic_network = Critic()

        # build up the target network
        self.actor_target_network = ActorSimple(self.num_edges,self.num_paths*num_request,self.num_paths)
        self.critic_target_network = Critic()

        # TODO:load weights into target
        # self.actor_network.variables

        # initialize optimizers
        lr_actor, lr_critic = 0.001, 0.0001
        self.actor_optim = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(lr_critic)

        # TODO:for saving model

    def _soft_update_target_network(self):
        for target_variable, variable in zip(self.actor_target_network.variables, self.actor_network.variables):
            target_variable.assign((1 - self.tau) * target_variable + self.tau * variable)
        for target_variable, variable in zip(self.critic_target_network.variables, self.critic_network.variables):
            target_variable.assign((1 - self.tau) * target_variable + self.tau * variable)

    def train(self, transitions, other_agents):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d_%d' % (self.src, self.dst)]  # 训练时只需要自己的reward
        u = []  # 用来装每个agent经验中的各项
        for i in range(self.args.num_nodes):
            for j in range(self.args.num_nodes):
                if i == j: continue
                u.append(transitions['u_%d_%d' % (i, j)])
        o = transitions['o']
        o_next = transitions['o_next']

        # calculate the target Q value function
        u_next = []
        with tf.GradientTape() as tape:
            # 得到下一个状态对应的动作
            index = 0
            for i in range(self.args.num_nodes):
                for j in range(self.args.num_nodes):
                    if i == j: continue
                    if i == self.src and j == self.dst:
                        u_next.append(self.actor_target_network(o_next))
                    else:
                        u_next.append(other_agents[(i, j)].policy.actor_target_network(o_next))
            print(u_next)
            state_next = DataProcesser.actions_to_state(o_next, u_next)
            # let data processor to do this
            q_next = self.critic_target_network(state_next)

            target_q = (r + self.args.gamma * q_next)

            # the q loss
            # let data processor to do this
            state = DataProcesser.actions_to_state(o, u)
            q_value = self.critic_network(state)
            critic_loss = tf.reduce_mean(np.power(target_q - q_value,2))
            if self.src < self.dst:
                idx = self.src*(self.args.num_nodes-1)+self.dst
            else:
                idx = self.src * (self.args.num_nodes - 1) + self.dst+1
            u[idx] = self.actor_network(o)
            state_after = DataProcesser.actions_to_state(o, u)
            actor_loss = - tf.reduce_mean(self.critic_network(state_after))

        # back propagation ac net
        actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
        print(actor_loss)
        print(actor_grad)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor_network.trainable_variables))
        critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
        self.critic_optim.apply_gradients(zip(critic_grad, self.critic_network.trainable_variables))

        # update target net
        self._soft_update_target_network()

    def save_model(self):
        pass

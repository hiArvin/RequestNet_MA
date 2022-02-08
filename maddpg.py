import tensorflow as tf
import numpy as np
from data_processor import DataProcesser
from actor_critic import Actor, Critic


class MADDPG:
    def __init__(self,args, src, dst):
        self.args = args
        self.src = src
        self.dst = dst
        self.num_paths = args.num_paths

        self.tau = 1.0 - 1e-2

        paths,idxs,seqs = DataProcesser.get_paths_inputs(src,dst)
        # create network
        self.actor_network = Actor(args.num_paths,paths,idxs,seqs)
        self.critic_network = Critic()

        # build up the target network
        self.actor_target_network = Actor(args.num_paths,paths,idxs,seqs)
        self.critic_target_network = Critic()

        # TODO:load weights into target
        # self.actor_network.variables

        # TODO:create the optimizer
        lr_actor, lr_critic = 0.001, 0.0001
        self.actor_optim = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(lr_critic)


        # TODO:for saving model

    def _soft_update_target_network(self):
        for target_variable,variable in zip(self.actor_target_network.variables,self.actor_network.variables):
            target_variable.assign((1 - self.tau) * target_variable + self.tau * variable)
        for target_variable,variable in zip(self.critic_target_network.variables,self.critic_network.variables):
            target_variable.assign((1 - self.tau) * target_variable + self.tau * variable)


    def train(self, transitions, other_agents):
        # for key in transitions.keys():
        #     transitions[key] = torch.tensor(transitions[key], dtype=torch.float32)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # calculate the target Q value function
        u_next = []
        with tf.GradientTape() as tape:
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1

            # TODO: processing u next to critic input
            # let data processor to do this
            q_next = self.critic_target_network(o_next, u_next)

            target_q = (r + self.args.gamma * q_next)

        # the q loss
        # let data processor to do this
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()

        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = - self.critic_network(o, u).mean()

        # back propagation ac net
        actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_weights)
        self.actor_optim.apply_gradients(zip(actor_grad, self.actor_network.trainable_weights))
        critic_grad = tape.gradient(critic_loss,self.critic_network.trainable_variables)
        self.critic_optim.apply_gradients(zip(critic_grad,self.critic_network.trainable_variables))

        # update target net
        self._soft_update_target_network()


    def save_model(self):
        pass

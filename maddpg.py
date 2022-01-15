import tensorflow as tf
from data_processer import DataProcesser
from actor_critic import Actor, Critic

class MADDPG:
    def __init__(self,agent_id,src,dst,paths):
        self.agent_id = agent_id
        self.src = src
        self.dst = dst

        # create network
        self.actor_network = Actor()
        self.critic_network = Critic()

        # build up the target network
        self.actor_target_network = Actor()
        self.critic_target_network = Critic()

        # TODO:load weights into target
        # self.actor_network.variables

        # TODO:create the optimizer
        lr_actor, lr_critic = 0.001, 0.0001
        self.actor_optim = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(lr_critic)

        # TODO:for saving model





    def path_preprocess(self):
        pass

    def _soft_update_target_network(self):
        pass

    def train(self):
        pass

    def save_model(self):
        pass

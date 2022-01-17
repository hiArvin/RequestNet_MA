import tensorflow as tf
from data_processer import DataProcesser
from actor_critic import Actor, Critic
from replay_buffer import ReplayBuffer


class MADDPG:
    def __init__(self,args, num_paths,agent_id, src, dst):
        # self.args = args
        self.agent_id = agent_id
        self.src = src
        self.dst = dst
        self.num_paths = num_paths
        paths,idxs,seqs = DataProcesser.get_paths_inputs(src,dst)
        # create network
        self.actor_network = Actor(num_paths,paths,idxs,seqs)
        self.critic_network = Critic()

        # build up the target network
        self.actor_target_network = Actor(num_paths,paths,idxs,seqs)
        self.critic_target_network = Critic()

        # TODO:load weights into target
        # self.actor_network.variables

        # TODO:create the optimizer
        lr_actor, lr_critic = 0.001, 0.0001
        self.actor_optim = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optim = tf.keras.optimizers.Adam(lr_critic)

        # Create experience buffer
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None

        # TODO:for saving model

    def _soft_update_target_network(self):
        pass

    def train(self):
        pass

    def save_model(self):
        pass

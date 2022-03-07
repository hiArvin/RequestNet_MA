import numpy as np
from env import Environment

env = Environment(3)
obs = env.reset()
print(obs)

actions ={}
for i in range(env.num_nodes):
    for j in range(env.num_nodes):
        actions[(i,j)]=np.array([0.5,0.3,0.2])

obs,rewards,_= env.step(actions)
print(obs)
print(rewards)

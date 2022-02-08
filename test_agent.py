import numpy as np
from env import Environment


from agent import Agent
import argparse

args = argparse.Namespace(num_paths=3)
env = Environment(args.num_paths)

obs,_,_ = env.reset()
print(obs)

# actions ={}
# for i in range(env.num_nodes):
#     for j in range(env.num_nodes):
#         actions[(i,j)]=np.array([0.5,0.3,0.2])
#
# obs,reward,_= env.step(actions)
# print(obs)
# print(reward)

agents = {}

for i in range(env.num_nodes):
    for j in range(env.num_nodes):
        if i==j:
            continue
        agent = Agent(i*env.num_nodes+j,args,i,j)
        agents[(i,j)]=agent

actions = {}
for i in range(env.num_nodes):
    for j in range(env.num_nodes):
        if i==j:
            continue
        actions[(i,j)]=agents[(i,j)].select_action(np.expand_dims(obs,axis=0),0.1)

obs,reward,_= env.step(actions)
print(obs)
print(reward)
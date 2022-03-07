import numpy as np
from maddpg.actor_critic import RequestNet, Critic

pf = np.array([[[0.265, 0.26, 0.24, 0.28, 0.29, 0.29, 0.285],
                [0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.]],

               [[0.265, 0.26, 0.24, 0.28, 0.29, 0.29, 0.285],
                [0.43, 0.47, 0.46, 0.505, 0.51, 0.52, 0.53],
                [0.265, 0.26, 0.24, 0.28, 0.29, 0.29, 0.285]]], dtype=np.float32)

print(pf.shape)  # num path , max len , link state
df = np.array([0.45, 0.59])

pf1 = np.expand_dims(pf, axis=0)
df1 = np.expand_dims(df, axis=0)

actor = RequestNet(num_paths=2, path_state_dim=14)


pfc = np.array([pf, pf,pf])
print(pfc.shape)
dfc = np.array([df, df,df])

outs = actor([pfc, dfc])
print(outs)
# outs= np.squeeze(outs)

# print(dfc.shape)
# outsc = np.array([outs, outs])
# print(outsc.shape)
#
# pfc1 = np.expand_dims(pfc,axis=0)
# dfc1 = np.expand_dims(dfc,axis=0)
# outsc1 = np.expand_dims(outsc,axis=0)
#
# critic = Critic(num_agents=2, num_paths=2, max_len=3,link_state_dim=7,path_state_dim=3)
# c_out = critic([pfc1, dfc1, outsc1])
# print(c_out)

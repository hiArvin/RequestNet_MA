import tensorflow as tf

from layers import PathEmbedding,FlowPointer
from actor_critic import Actor


paths = [0,1,3,2]
index = [0,0,1,1]
sequences =[0,1,0,1]


actor = Actor(src=1, dst=1, num_paths=2, paths=paths, idx=index, seq=sequences, theta1=2, theta2=2, theta3=2)

x = tf.ones((4,2))

y = actor(x)

target_actor = Actor(src=1, dst=1, num_paths=2, paths=paths, idx=index, seq=sequences, theta1=2, theta2=2, theta3=2)

for layer in target_actor.layers:
    target_actor.variables = actor.variables
print(actor.trainable_variables)
print(y)
# y = my_layer1(x)
# # z = my_layer2(y)
# print(y.numpy())
# print(z.numpy())

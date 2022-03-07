from maddpg.layers import PathEmbedding,FlowPointer
import  tensorflow as tf
paths = [0,1,3,2,1]
index = [0,0,1,1,1]
sequences =[0,1,0,1,2]
l1 = PathEmbedding(num_paths=2,
                 path_state_dim=3,
                 paths=paths,
                 index=index,
                 sequences=sequences)
l2 = FlowPointer(2)
x = tf.ones((5,4,4))
y = l1(x)
z = l2(y)
print(y)
print(z)
print(l2.variables)

from layers import PathEmbedding,FlowPointer
import tensorflow as tf

paths = [0,1,3,2]
index = [0,0,1,1]
sequences =[0,1,0,1]

my_layer1 = PathEmbedding(num_paths=2,
                         path_state_dim=3,
                         paths=paths,
                         index=index,
                         sequences=sequences)
my_layer2 = FlowPointer(2,2)

x = tf.ones((4,2))

y = my_layer1(x)
z = my_layer2(y)
print(y.numpy())
print(z.numpy())

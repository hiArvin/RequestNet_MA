from layers import PathEmbedding
import tensorflow as tf

paths = [0,1,3,2]
index = [0,0,1,1]
sequences =[0,1,0,1]

my_layer = PathEmbedding(num_paths=2,
                         num_edges=4,
                         link_state_dim=1,
                         path_state_dim=1,
                         paths=paths,
                         index=index,
                         sequences=sequences)


x = tf.ones((4,1))

y = my_layer(x)

print(y.numpy())
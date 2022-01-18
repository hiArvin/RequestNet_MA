from layers import PathEmbedding
import  tensorflow as tf
paths = [0,1,3,2]
index = [0,0,1,1]
sequences =[0,1,0,1]
l1 = PathEmbedding(num_paths=2,
                 path_state_dim=2,
                 paths=paths,
                 index=index,
                 sequences=sequences)
x = tf.ones((4,2))
l1(x)
print(l1.get_weights())
import tensorflow as tf
import numpy as np
batch_size = 10
idx = [1,3,4,5,2]

batch = tf.range(batch_size)
batch = tf.tile(tf.expand_dims(batch,1),(1,len(idx)))
# batch = tf.reshape(batch,batch_size*len(idx))
print(batch)
import tensorflow as tf
import numpy as np
from gqn.gqn_graph import *

img = tf.placeholder(shape=(1, 64, 64, 3), dtype=tf.float32)
pose = tf.placeholder(shape=(1,7), dtype=tf.float32)

representation, _ = gqn(img, pose)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    a = s.run(representation, feed_dict={
                img: np.random.uniform(size=(1, 64, 64, 3)),
                pose: np.random.uniform(size=(1, 7)),
              })

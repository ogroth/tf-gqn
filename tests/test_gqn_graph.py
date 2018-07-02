"""
Quick test script to check graph definition of GQN.
"""

import tensorflow as tf
import numpy as np

from gqn.gqn_graph import gqn

# graph definition
img = tf.placeholder(shape=(1, 64, 64, 3), dtype=tf.float32)
pose = tf.placeholder(shape=(1,7), dtype=tf.float32)
representation, _ = gqn(img, pose)

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  a = sess.run(
      representation,
      feed_dict={
          img: np.random.uniform(size=(1, 64, 64, 3)),
          pose: np.random.uniform(size=(1, 7)),
      })

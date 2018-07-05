"""
Quick test script to check graph definition of full GQN model.
"""

import tensorflow as tf
import numpy as np

from gqn.gqn_params import PARAMS
from gqn.gqn_graph import gqn

# constants
_BATCH_SIZE = 1
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_H_IMG = PARAMS.IMG_HEIGHT
_DIM_W_IMG = PARAMS.IMG_WIDTH
_DIM_C_IMG = PARAMS.IMG_CHANNELS

# graph definition
img = tf.placeholder(shape=[_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG], dtype=tf.float32)
pose = tf.placeholder(shape=[_BATCH_SIZE, _DIM_POSE], dtype=tf.float32)
net, endpoints = gqn(img, pose)

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  a = sess.run(
      net,
      feed_dict={
          img: np.random.uniform(size=[_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG]),
          pose: np.random.uniform(size=[_BATCH_SIZE, _DIM_POSE]),
      })

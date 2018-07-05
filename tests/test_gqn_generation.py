"""
Quick test script to check graph definition of GQN image generation.
"""

import tensorflow as tf
import numpy as np

from gqn.gqn_params import PARAMS
from gqn.gqn_rnn import generator_rnn

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = 5
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_R_H = 1
_DIM_R_W = 1
_DIM_R_C = PARAMS.REPRESENTATION_CHANNELS
_SEQ_LENGTH = 1

# input placeholders
query_pose = tf.placeholder(
    shape=(_BATCH_SIZE, _DIM_POSE), dtype=tf.float32)
scene_representation = tf.placeholder(
    shape=(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C), dtype=tf.float32)

# set up the generator LSTM cell
canvas = generator_rnn(
    poses=query_pose,
    representations=scene_representation,
    sequence_size=_SEQ_LENGTH)

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  feed_dict = {
      query_pose : np.random.rand(_BATCH_SIZE, _DIM_POSE),
      scene_representation : np.random.rand(_BATCH_SIZE, _DIM_R_H, _DIM_R_W, _DIM_R_C),
  }  
  c = sess.run(canvas,feed_dict=feed_dict)
  print(c)

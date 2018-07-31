"""
Quick test script to shape-check graph definition of GQN encoder with random
toy data.
"""

import os
import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, '..'))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import numpy as np

from gqn.gqn_params import PARAMS
from gqn.gqn_encoder import pool_encoder

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = 5
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_H_IMG = PARAMS.IMG_HEIGHT
_DIM_W_IMG = PARAMS.IMG_WIDTH
_DIM_C_IMG = PARAMS.IMG_CHANNELS
_DIM_C_ENC = PARAMS.ENC_CHANNELS

# input placeholders
context_poses = tf.placeholder(
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE],
    dtype=tf.float32)
context_frames = tf.placeholder(
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG],
    dtype=tf.float32)

# reshape context pairs into pseudo batch for representation network
context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
context_frames_packed = tf.reshape(context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

# set up encoder for scene representation
r_encoder_batch, ep_encoding = pool_encoder(context_frames_packed, context_poses_packed)
r_encoder_batch = tf.reshape(
    r_encoder_batch,
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, 1, 1, _DIM_C_ENC])  # 1, 1 for pool encoder only!
r_encoder = tf.reduce_sum(r_encoder_batch, axis=1) # add scene representations per data tuple

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  feed_dict = {
      context_poses : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE),
      context_frames : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
  }  
  r = sess.run(r_encoder, feed_dict=feed_dict)
  print(r)
  print(r.shape)
  for ep, t in ep_encoding.items():
    print(ep, t)

print("TEST PASSED!")

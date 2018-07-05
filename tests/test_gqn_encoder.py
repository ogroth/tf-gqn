"""
Quick test script to check graph definition of GQN.
"""

import tensorflow as tf
import numpy as np

from gqn.gqn_params import PARAMS
from gqn.gqn_graph import gqn
from gqn.gqn_encoder import pool_encoder

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = 5
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_H_IMG = PARAMS.IMG_HEIGHT
_DIM_W_IMG = PARAMS.IMG_WIDTH
_DIM_C_IMG = PARAMS.IMG_CHANNELS
_DIM_R = PARAMS.REPRESENTATION_CHANNELS

# input placeholders
query_pose = tf.placeholder(
    shape=(_BATCH_SIZE, _DIM_POSE), dtype=tf.float32)
target_frame = tf.placeholder(
    shape=(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
    dtype=tf.float32)
context_poses = tf.placeholder(
    shape=(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE),
    dtype=tf.float32)
context_frames = tf.placeholder(
    shape=(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
    dtype=tf.float32)

# reshape context pairs into pseudo batch for representation network
context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
context_frames_packed = tf.reshape(context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

# set up encoder for scene representation
r_encoder_batch, ep_encoder = pool_encoder(context_frames_packed, context_poses_packed)
r_encoder_batch = tf.reshape(
    r_encoder_batch,
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, 1, 1, _DIM_R])
r_encoder = tf.reduce_sum(r_encoder_batch, axis=1) # add scene representations per data tuple

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  feed_dict = {
    query_pose : np.random.rand(_BATCH_SIZE, _DIM_POSE),
    target_frame : np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
    context_poses : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE),
    context_frames : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
  }  
  r = sess.run(r_encoder,feed_dict=feed_dict)
  print(r)

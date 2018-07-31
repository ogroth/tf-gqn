"""
Quick test script to shape-check graph definition of full GQN model with random
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
from gqn.gqn_graph import gqn_draw, gqn_vae

# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = PARAMS.CONTEXT_SIZE
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_H_IMG = PARAMS.IMG_HEIGHT
_DIM_W_IMG = PARAMS.IMG_WIDTH
_DIM_C_IMG = PARAMS.IMG_CHANNELS
_SEQ_LENGTH = PARAMS.SEQ_LENGTH

# input placeholders
query_pose = tf.placeholder(
    shape=[_BATCH_SIZE, _DIM_POSE], dtype=tf.float32)
target_frame = tf.placeholder(
    shape=[_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG],
    dtype=tf.float32)
context_poses = tf.placeholder(
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE],
    dtype=tf.float32)
context_frames = tf.placeholder(
    shape=[_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG],
    dtype=tf.float32)

# graph definition
net, ep_gqn = gqn_draw(
    query_pose=query_pose,
    target_frame=target_frame,
    context_poses=context_poses,
    context_frames=context_frames,
    model_params=PARAMS,
    is_training=True
)

net_vae, ep_gqn_vae = gqn_vae(
  query_pose=query_pose,
  context_poses=context_poses,
  context_frames=context_frames,
  model_params=PARAMS,
)

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  [mu, mu_vae] = sess.run(
      [net, net_vae],
      feed_dict={
          query_pose : np.random.rand(_BATCH_SIZE, _DIM_POSE),
          target_frame : np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
          context_poses : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE),
          context_frames : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
      })
  print(mu, mu_vae)
  print(mu.shape, mu_vae.shape)

  for ep, t in ep_gqn.items():
    print(ep, t)

  for ep, t in ep_gqn_vae.items():
    print(ep, t)

print("TEST PASSED!")

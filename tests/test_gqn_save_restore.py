"""
Quick test script to check correct save and restore functionality of GQN model.
"""

import os
import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, '..'))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import numpy as np

from gqn.gqn_params import GQN_DEFAULT_CONFIG
from gqn.gqn_graph import gqn_draw


# constants
_BATCH_SIZE = 1
_CONTEXT_SIZE = GQN_DEFAULT_CONFIG.CONTEXT_SIZE
_DIM_POSE = GQN_DEFAULT_CONFIG.POSE_CHANNELS
_DIM_H_IMG = GQN_DEFAULT_CONFIG.IMG_HEIGHT
_DIM_W_IMG = GQN_DEFAULT_CONFIG.IMG_WIDTH
_DIM_C_IMG = GQN_DEFAULT_CONFIG.IMG_CHANNELS
_SEQ_LENGTH = GQN_DEFAULT_CONFIG.SEQ_LENGTH


def create_input_placeholders():
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

  return query_pose, target_frame, context_poses, context_frames


# input placeholders
query_pose, target_frame, context_poses, context_frames = \
    create_input_placeholders()

# graph definition in training mode
net, ep_gqn = gqn_draw(
    query_pose=query_pose,
    target_frame=target_frame,
    context_poses=context_poses,
    context_frames=context_frames,
    model_params=GQN_DEFAULT_CONFIG,
    is_training=True
)

# print computational endpoints
print("GQN enpoints:")
for ep, t in ep_gqn.items():
  print(ep, t)

# feed the graph with a random input data point
rnd_query_pose = np.random.rand(_BATCH_SIZE, _DIM_POSE)
rnd_target_frame = np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)
rnd_context_poses = np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE)
rnd_context_frames = np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG)

saver = tf.train.Saver()
with tf.Session() as sess:
  # Run Initializers
  sess.run(tf.global_variables_initializer())

  # Run network forward
  sess.run(
      net,
      feed_dict={
          query_pose: rnd_query_pose,
          target_frame: rnd_target_frame,
          context_poses: rnd_context_poses,
          context_frames: rnd_context_frames,
      })
  saver.save(sess, save_path="/tmp/gqn_test_checkpoint")

print("Saved model!")

# Reset everything
tf.reset_default_graph()

# input placeholders
query_pose, target_frame, context_poses, context_frames = \
  create_input_placeholders()

# graph definition in test mode
net, ep_gqn = gqn_draw(
    query_pose=query_pose,
    target_frame=target_frame,
    context_poses=context_poses,
    context_frames=context_frames,
    model_params=GQN_DEFAULT_CONFIG,
    is_training=False
)


saver = tf.train.Saver()
with tf.Session() as sess:
  # Don't run initalisers, restore variables instead
  # sess.run(tf.global_variables_initializer())
  saver.restore(sess, save_path="/tmp/gqn_test_checkpoint")

  # Run network forward, shouldn't complain about uninitialised variables
  sess.run(
      net,
      feed_dict={
          query_pose: rnd_query_pose,
          target_frame: rnd_target_frame,
          context_poses: rnd_context_poses,
          context_frames: rnd_context_frames,
      })
print("Restored variables and ran a forward pass... Yay!")

print("TEST PASSED!")

"""
Quick test script to check graph definition of full GQN model.
"""

import tensorflow as tf
import numpy as np

from gqn.gqn_params import PARAMS
from gqn.gqn_graph import gqn_draw
from gqn.gqn_objective import gqn_draw_elbo

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

# loss definition
mu_target = net
sigma_target = tf.constant(  # additional parameter tuned during training
    value=1.0, dtype=tf.float32,
    shape=[_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
# collecting endpoints for ELBO computation
for i in range(_SEQ_LENGTH):
  mu_q.append(ep_gqn["mu_q_%d" % i])
  sigma_q.append(ep_gqn["sigma_q_%d" % i])
  mu_pi.append(ep_gqn["mu_pi_%d" % i])
  sigma_pi.append(ep_gqn["sigma_pi_%d" % i])
elbo = gqn_draw_elbo(
    mu_target, sigma_target,
    mu_q, sigma_q,
    mu_pi, sigma_pi,
    target_frame)

# feed random input through the graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  loss = sess.run(
      elbo,
      feed_dict={
          query_pose : np.random.rand(_BATCH_SIZE, _DIM_POSE),
          target_frame : np.random.rand(_BATCH_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
          context_poses : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_POSE),
          context_frames : np.random.rand(_BATCH_SIZE, _CONTEXT_SIZE, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
      })
  print(loss)
  print(loss.shape)
  for ep, t in ep_gqn.items():
    print(ep, t)

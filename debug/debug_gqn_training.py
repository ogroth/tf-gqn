"""
Debugging script (feed_dict loop) for GQN training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse

import tensorflow as tf

# assumes tf-gqn path to live in PYTHONPATH!
from gqn.gqn_params import PARAMS
from gqn.gqn_graph import gqn
from gqn.gqn_objective import gqn_elbo
from data_provider.gqn_tfr_provider import DataReader


# constants
# TODO(ogroth): to be replaced with parameter set loaded from config
_CONTEXT_SIZE = PARAMS.CONTEXT_SIZE
_DIM_POSE = PARAMS.POSE_CHANNELS
_DIM_H_IMG = PARAMS.IMG_HEIGHT
_DIM_W_IMG = PARAMS.IMG_WIDTH
_DIM_C_IMG = PARAMS.IMG_CHANNELS
_SEQ_LENGTH = PARAMS.SEQ_LENGTH


# CLI arguments
ARGPARSER = argparse.ArgumentParser(
    description='Debug training run for GQN without tf.estimator.')
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/Volumes/Maxtor/datasets/gqn-dataset',
    help='The path to the gqn-dataset directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='rooms_ring_camera',
    help='The name of the GQN dataset to use. \
    Available names are: \
    jaco | mazes | rooms_free_camera_no_object_rotations | \
    rooms_free_camera_with_object_rotations | rooms_ring_camera | \
    shepard_metzler_5_parts | shepard_metzler_7_parts')
ARGPARSER.add_argument(
    '--batch_size', type=int, default=12,
    help='The number of data points per batch. One data point is a tuple of \
    ((query_camera_pose, [(context_frame, context_camera_pose)]), target_frame).')
ARGPARSER.add_argument(
    '--steps_train', type=int, default=10000,
    help='The number of parameter updates to perform.')


if __name__ == '__main__':
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("Debug training run of GQN for %s steps." % (FLAGS.steps_train, ))
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)

  # data reader
  data_reader = DataReader(
      dataset=FLAGS.dataset,
      context_size=_CONTEXT_SIZE,
      root=FLAGS.data_dir)
  data = data_reader.read(batch_size=FLAGS.batch_size)

  # map data tuple to model inputs
  query_pose = data.query.query_camera
  target_frame = data.target
  context_poses = data.query.context.cameras
  context_frames = data.query.context.frames

  # graph definition
  net, ep_gqn = gqn(
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
      shape=[FLAGS.batch_size, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])
  # collecting endpoints for ELBO computation
  mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
  for i in range(_SEQ_LENGTH):
    mu_q.append(ep_gqn["mu_q_%d" % i])
    sigma_q.append(ep_gqn["sigma_q_%d" % i])
    mu_pi.append(ep_gqn["mu_pi_%d" % i])
    sigma_pi.append(ep_gqn["sigma_pi_%d" % i])
  elbo = gqn_elbo(
      mu_target, sigma_target,
      mu_q, sigma_q,
      mu_pi, sigma_pi,
      target_frame)

  # optimizer
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(loss=elbo)

  # print computational endpoints
  print("GQN enpoints:")
  for ep, t in sorted(ep_gqn.items(), key=lambda x: x[0]):
    print(ep, t)

  # training loop
  with tf.train.SingularMonitoredSession() as sess:
    for step in range(FLAGS.steps_train):
      # run one forward pass, compute elbo and update weights
      _elbo, _train_op = sess.run([elbo, train_op])
      print("Training step: %d" % (step + 1, ))
      print("Negative ELBO: %f" % (_elbo, ))

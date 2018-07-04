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

# assumes tf-gqn path to live in PYTHONPATH
from gqn.gqn_graph import pool_encoder
from data_provider.gqn_tfr_provider import DataReader

# constants
# TODO(ogroth): refactor model constants into gqn module
_DIM_POSE = 7
_DIM_H_IMG = 64
_DIM_W_IMG = 64
_DIM_C_IMG = 3

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
    '--context_size', type=int, default=5,
    help='The number of (frame, camera_pose) pairs provided as a context of \
    a query pose.')
ARGPARSER.add_argument(
    '--batch_size', type=int, default=1, # 12
    help='The number of data points per batch. One data point is a tuple of \
    ((query_camera_pose, [(context_frame, context_camera_pose)]), target_frame).')
ARGPARSER.add_argument(
    '--steps_train', type=int, default=1,
    help='The number of parameter updates to perform.')


if __name__ == '__main__':
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("Debug training run of GQN for %s steps." % (FLAGS.steps_train, ))
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)

  # data reader
  data_reader = DataReader(
      dataset=FLAGS.dataset,
      context_size=FLAGS.context_size,
      root=FLAGS.data_dir)
  data = data_reader.read(batch_size=FLAGS.batch_size)

  # input placeholders
  query_pose = tf.placeholder(
      shape=(FLAGS.batch_size, _DIM_POSE), dtype=tf.float32)
  target_frame = tf.placeholder(
      shape=(FLAGS.batch_size, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
      dtype=tf.float32)
  context_poses = tf.placeholder(
      shape=(FLAGS.batch_size, FLAGS.context_size, _DIM_POSE),
      dtype=tf.float32)
  context_frames = tf.placeholder(
      shape=(FLAGS.batch_size, FLAGS.context_size, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG),
      dtype=tf.float32)

  # reshape context pairs into pseudo batch for representation network
  context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
  context_frames_packed = tf.reshape(context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

  # set up encoder for scene representation
  r_encoder_batch, ep_encoder = pool_encoder(context_frames_packed, context_poses_packed)
  r_encoder_batch = tf.reshape(
      r_encoder_batch,
      shape=[FLAGS.batch_size, FLAGS.context_size, 1, 1, 256]) # TODO(ogroth): parameterize reshape!
  r_encoder = tf.reduce_sum(r_encoder_batch, axis=1) # add scene representations per data tuple

  # loss function
  # STUB

  # optimizer
  # STUB

  # training loop
  with tf.train.SingularMonitoredSession() as sess:
    for s in range(FLAGS.steps_train):
      d = sess.run(data) # runs the dequeue ops to fetch the data
      # decompose data tuple into feed_dict
      feed_dict = {
        query_pose : d.query.query_camera,
        target_frame : d.target,
        context_poses : d.query.context.cameras,
        context_frames : d.query.context.frames
      }
      r = sess.run(r_encoder, feed_dict=feed_dict)
      print(r)

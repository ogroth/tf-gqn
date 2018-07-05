"""
Contains utility functions for the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .gqn_params import PARAMS


def broadcast_poses(poses, height, width):
  """
  Broadcasts a pose vector to every pixel of an image.
  """
  # with tf.control_dependencies([tf.assert_rank(poses, 2)]):
  poses = tf.reshape(poses, [-1, 1, 1, PARAMS.POSE_CHANNELS])
  poses = tf.tile(poses, [1, height, width, 1])
  return poses

def eta(h, kernel_size=5, scope=None):
  with tf.variable_scope(scope):
    # TODO(stefan,ogroth): what's the activation function??
    ita = tf.layers.conv2d(h, filters=2 * PARAMS.Z_CHANNELS, kernel_size=kernel_size,
                           padding='SAME')
    mu, sigma = tf.split(ita, num_or_size_splits=2, axis=-1)

    return mu, sigma

def sample_z(h, kernel_size=5, scope=None):
  mu, sigma = eta(h, kernel_size, scope)

  with tf.variable_scope(scope):
    z_shape = tf.shape(h).as_list()[:-1] + [PARAMS.Z_CHANNELS]
    z = mu + tf.multiply(sigma, tf.random_normal(shape=z_shape))

    return z

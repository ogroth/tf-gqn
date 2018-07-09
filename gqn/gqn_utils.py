"""
Contains utility functions for the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .gqn_params import PARAMS


def add_scope(fn, scope):
  def _wrapper(*args, **kwargs):
    if scope is not None:
      with tf.variable_scope(scope):
        return fn(*args, **kwargs)
    else:
      return fn(*args, **kwargs)

  return _wrapper


def optional_scope_default(default_scope=None):
    def _optional_scope(fn):
        def extract_and_add_scope(*args, **kwargs):
            scope = kwargs.pop("scope", default_scope)
            return add_scope(fn, scope)(*args, **kwargs)

        return extract_and_add_scope

    return _optional_scope


optional_scope = optional_scope_default(None)


def create_sub_scope(scope, name):
  if scope is None:
    varscope = tf.get_variable_scope()
  else:
    varscope = tf.variable_scope(scope)

  with varscope, tf.variable_scope(name) as subvarscope:
    return subvarscope


def broadcast_poses(poses, height, width):
  """
  Broadcasts a pose vector to every pixel of an image.
  """
  poses = tf.reshape(poses, [-1, 1, 1, PARAMS.POSE_CHANNELS])
  poses = tf.tile(poses, [1, height, width, 1])
  return poses


@optional_scope
def eta(h, kernel_size=5):
  """
  Computes sufficient statistics of a normal distribution (mu, sigma) from a hidden state
  representation via convolution.
  """
  # TODO(stefan,ogroth): activation not specified in the paper, guess: linear
  eta = tf.layers.conv2d(h, filters=2*PARAMS.Z_CHANNELS, kernel_size=kernel_size,
                         padding='SAME')
  mu, sigma = tf.split(eta, num_or_size_splits=2, axis=-1)

  return mu, sigma


@optional_scope
def compute_eta_and_sample_z(h, kernel_size=5):
  """
  Samples a variational encoding vector z from a normal distribution parameterized by a hidden
  state h.
  Statistics of the normal distribution are obtained from h.
  The sampling is done via the 're-parameterization trick' (factoring out noise into epsilon).
  """
  mu, sigma = eta(h, kernel_size, scope="eta")
  with tf.variable_scope("Sampling"):
    z_shape = tf.concat([tf.shape(h)[:-1], [PARAMS.Z_CHANNELS]], axis=0,
                        name="CreateZShape")
    z = mu + tf.multiply(sigma, tf.random_normal(shape=z_shape))

  return mu, sigma, z


@optional_scope
def sample_z(h, kernel_size=5):
  _, _, z = compute_eta_and_sample_z(h, kernel_size)
  return z

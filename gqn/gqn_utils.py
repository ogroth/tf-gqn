"""
Contains utility functions for the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .gqn_params import PARAMS


def add_scope(fn, scope):
  # TODO(stefan): add docstring
  def _wrapper(*args, **kwargs):
    if scope is not None:
      with tf.variable_scope(scope):
        return fn(*args, **kwargs)
    else:
      return fn(*args, **kwargs)

  return _wrapper


def optional_scope_default(default_scope=None):
  # TODO(stefan): add docstring
    def _optional_scope(fn):
        def extract_and_add_scope(*args, **kwargs):
            scope = kwargs.pop("scope", default_scope)
            return add_scope(fn, scope)(*args, **kwargs)

        return extract_and_add_scope

    return _optional_scope


optional_scope = optional_scope_default(None)


def create_sub_scope(scope, name):
  # TODO(stefan): add docstring
  if scope is None:
    varscope = tf.get_variable_scope()
  else:
    varscope = tf.variable_scope(scope)

  with varscope, tf.variable_scope(name) as subvarscope:
    return subvarscope


def broadcast_pose(vector, height, width):
  """
  Broadcasts a pose vector to every pixel of a target image.
  """
  vector = tf.reshape(vector, [-1, 1, 1, PARAMS.POSE_CHANNELS])
  vector = tf.tile(vector, [1, height, width, 1])
  return vector

def broadcast_encoding(vector, height, width):
  """
  Broadcasts a scene encoding to every pixel of a target image.
  """
  vector = tf.reshape(vector, [-1, 1, 1, PARAMS.ENC_CHANNELS])
  vector = tf.tile(vector, [1, height, width, 1])
  return vector


@optional_scope
def eta(h, kernel_size=PARAMS.LSTM_KERNEL_SIZE, channels=PARAMS.Z_CHANNELS):
  """
  Computes sufficient statistics of a normal distribution (mu, sigma) from a hidden state
  representation via convolution.
  """
  # TODO(stefan,ogroth): activation not specified in the paper, guess: linear
  eta = tf.layers.conv2d( # 2 * channels because mu and sigma need to be computed per channel
      h, filters=2*channels, kernel_size=kernel_size, padding='SAME')
  mu, sigma = tf.split(eta, num_or_size_splits=2, axis=-1)
  sigma = tf.nn.elu(sigma) + tf.constant(1.0, dtype=tf.float32)  # ensuring sigma > 0

  return mu, sigma


@optional_scope
def compute_eta_and_sample_z(h, kernel_size=PARAMS.LSTM_KERNEL_SIZE, channels=PARAMS.Z_CHANNELS):
  """
  Samples a variational encoding vector z from a normal distribution parameterized by a hidden
  state h.
  Statistics of the normal distribution are obtained from h via a convolutional function eta.
  The sampling is done via the 're-parameterization trick' (factoring out noise into epsilon).
  """
  mu, sigma = eta(h, kernel_size, channels, scope="eta")
  with tf.variable_scope("Sampling"):
    z_shape = tf.concat([tf.shape(h)[:-1], [channels]], axis=0,
                        name="CreateZShape")
    z = mu + tf.multiply(sigma, tf.random_normal(shape=z_shape))

  return mu, sigma, z


@optional_scope
def sample_z(h, kernel_size=PARAMS.LSTM_KERNEL_SIZE, channels=PARAMS.Z_CHANNELS):
  """
  Samples a variational encoding vector z from a normal distribution parameterized by a hidden
  state h.
  Statistics of the normal distribution are obtained from h via a convolutional function eta.
  The sampling is done via the 're-parameterization trick' (factoring out noise into epsilon).
  """
  _, _, z = compute_eta_and_sample_z(h, kernel_size)
  return z

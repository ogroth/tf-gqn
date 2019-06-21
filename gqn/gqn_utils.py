"""
Contains utility functions for the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .gqn_params import GQN_DEFAULT_CONFIG


# scoping utilities

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


# shaping utilities

def broadcast_pose(vector, height, width):
  """
  Broadcasts a pose vector to every pixel of a target image.
  """
  vector = tf.reshape(vector, [-1, 1, 1, GQN_DEFAULT_CONFIG.POSE_CHANNELS])
  vector = tf.tile(vector, [1, height, width, 1])
  return vector


def broadcast_encoding(vector, height, width):
  """
  Broadcasts a scene encoding to every pixel of a target image.
  """
  vector = tf.reshape(vector, [-1, 1, 1, GQN_DEFAULT_CONFIG.ENC_CHANNELS])
  vector = tf.tile(vector, [1, height, width, 1])
  return vector


# sampling utilities

def eta_g(canvas,
          kernel_size=GQN_DEFAULT_CONFIG.ETA_EXTERNAL_KERNEL_SIZE,
          channels=GQN_DEFAULT_CONFIG.IMG_CHANNELS,
          scope="eta_g"):
  return tf.layers.conv2d(
      canvas, filters=channels, kernel_size=kernel_size, padding='SAME',
      name=scope)


@optional_scope
def eta(h, kernel_size=GQN_DEFAULT_CONFIG.LSTM_KERNEL_SIZE, channels=GQN_DEFAULT_CONFIG.Z_CHANNELS):
  """
  Computes sufficient statistics of a normal distribution (mu, sigma) from a
  hidden state representation via convolution.
  """
  # TODO(stefan,ogroth): activation not specified in the paper, guess: linear
  eta = tf.layers.conv2d( # 2 * channels because mu and sigma need to be computed per channel
      h, filters=2*channels, kernel_size=kernel_size, padding='SAME')
  mu, sigma = tf.split(eta, num_or_size_splits=2, axis=-1)
  # sigma = tf.nn.elu(sigma) + 1.0  # ensuring sigma > 0
  sigma = tf.nn.softplus(sigma + .5) + 1e-8  # TODO(ogroth): check

  return mu, sigma


@optional_scope
def compute_eta_and_sample_z(h, kernel_size=GQN_DEFAULT_CONFIG.ETA_INTERNAL_KERNEL_SIZE, channels=GQN_DEFAULT_CONFIG.Z_CHANNELS):
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
def sample_z(h, kernel_size=GQN_DEFAULT_CONFIG.ETA_INTERNAL_KERNEL_SIZE, channels=GQN_DEFAULT_CONFIG.Z_CHANNELS):
  """
  Samples a variational encoding vector z from a normal distribution parameterized by a hidden
  state h.
  Statistics of the normal distribution are obtained from h via a convolutional function eta.
  The sampling is done via the 're-parameterization trick' (factoring out noise into epsilon).
  """
  _, _, z = compute_eta_and_sample_z(h, kernel_size, channels)
  return z


# visualization utilities

_BAR_WIDTH = 2


def debug_canvas_image_mean(canvases, eta_g_scope='GQN'):
  """
  Projects the canvas into mean images.
  """
  with tf.variable_scope(eta_g_scope, reuse=True, auxiliary_name_scope=False), \
      tf.name_scope('debug'):
    mean_images = []

    with tf.name_scope('MakeWhiteBar'):
      cs = tf.shape(canvases[0])
      batch, height, channels = cs[0], cs[1], 3
      white_vertical_bar = tf.ones(
          shape=(batch, height, _BAR_WIDTH, channels),
          dtype=tf.float32,
          name='white_bar')

    for canvas in canvases:
      mean_images.append(eta_g(canvas))
      mean_images.append(white_vertical_bar)

    return tf.concat(mean_images[:-1], axis=-2, name='canvas_grid')

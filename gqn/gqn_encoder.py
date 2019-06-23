"""
Contains the graph definition of the GQN encoding stack.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .gqn_utils import broadcast_pose


def tower_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="TowerEncoder"):
  """
  Feed-forward convolutional architecture.
  """
  with tf.variable_scope(scope):
    endpoints = {}
    net = tf.layers.conv2d(frames, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)
    skip1 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip1
    net = tf.layers.conv2d(net, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)

    # tile the poses to match the embedding shape
    height, width = tf.shape(net)[1], tf.shape(net)[2]
    poses = broadcast_pose(poses, height, width)

    # concatenate the poses with the embedding
    net = tf.concat([net, poses], axis=3)

    skip2 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip2

    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    return net, endpoints


def pool_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="PoolEncoder"):
  """
  Feed-forward convolutional architecture with terminal global pooling.
  """
  net, endpoints = tower_encoder(frames, poses, scope)
  with tf.variable_scope(scope):
    net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)

  return net, endpoints


def self_attention(layer: tf.Tensor, name: str):
  """
  Self-Attention as described in Self-Attention Generative Adversarial Networks by Zhang, Goodfellow, Metaxas, and Odena
  """
  m = tf.shape(layer)[0]
  _, X, Y, C = layer.shape
  W_f = tf.get_variable(f'W_{name}_f', shape=[C, C], initializer=tf.contrib.layers.xavier_initializer())
  f = tf.tensordot(layer, W_f, [[-1], [0]])
  W_g = tf.get_variable(f'W_{name}_g', shape=[C, C], initializer=tf.contrib.layers.xavier_initializer())
  g = tf.tensordot(layer, W_g, [[-1], [0]])
  W_h = tf.get_variable(f'W_{name}_h', shape=[C, C], initializer=tf.contrib.layers.xavier_initializer())
  h = tf.tensordot(layer, W_h, [[-1], [0]])
  f = tf.reshape(f, shape=[-1, X * Y * C])
  g = tf.reshape(g, shape=[-1, X * Y * C])
  s = tf.matmul(f, g, transpose_b=True)
  s = tf.reshape(s, shape=[-1, X * Y * C, X * Y * C])
  s = tf.nn.softmax(s, -1)
  o = tf.tensordot(s, tf.reshape(h, [-1, X*Y*C]), [[2], [1]])
  o = tf.reshape(h, [-1, X, Y, C])
  gamma = tf.get_variable(f'gamma_{name}', shape=[1, X, Y, C], initializer=tf.zeros_initializer())
  return gamma * o + layer


def sa_encoder(frames: tf.Tensor, poses: tf.Tensor, scope="SAEncoder"):
  """
  Feed-forward convolutional architecture with self-attention (modified from tower+pool to add self-attention.)
  """
  with tf.variable_scope(scope):
    endpoints = {}
    net = tf.layers.conv2d(frames, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)
    net = self_attention(net, "l1")
    skip1 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = net + skip1
    net = self_attention(net, "l2")
    net = tf.layers.conv2d(net, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)
    net = self_attention(net, "l3")

    # tile the poses to match the embedding shape
    height, width = tf.shape(net)[1], tf.shape(net)[2]
    poses = broadcast_pose(poses, height, width)

    # concatenate the poses with the embedding
    net = tf.concat([net, poses], axis=3)

    skip2 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = net + skip2
    net = self_attention(net, "l4")

    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = self_attention(net, "l5")

    net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = self_attention(net, "l6")

    net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)

  return net, endpoints

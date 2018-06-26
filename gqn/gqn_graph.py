"""
Contains the GQN graph definition.

Original paper:
'Neural scene representation and rendering'
S. M. Ali Eslami, Danilo J. Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos, 
Marta Garnelo, Avraham Ruderman, Andrei A. Rusu, Ivo Danihelka, Karol Gregor, 
David P. Reichert, Lars Buesing, Theophane Weber, Oriol Vinyals, Dan Rosenbaum, 
Neil Rabinowitz, Helen King, Chloe Hillier, Matt Botvinick, Daan Wierstra, 
Koray Kavukcuoglu and Demis Hassabis
https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def tower_encoder(images, poses, scope="TowerEncoder"):
  with tf.variable_scope(scope):
    endpoints = {}
    net = tf.layers.conv2d(images, filters=256, kernel_size=2, strides=2,
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
    with tf.control_dependencies([tf.assert_rank(poses, 2)]):  # (batch, 7)
        height, width = tf.shape(net)[1], tf.shape(net)[2]

        poses = tf.reshape(poses, [-1, 1, 1, 7])
        poses = tf.tile(poses, [1, height, width, 1])

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


def pool_encoder(images, poses, scope="PoolEncoder"):
  net, endpoints = tower_encoder(images, poses, scope)
  with tf.variable_scope(scope):
      net = tf.reduce_mean(net, axis=[1, 2], keepdims=True)

  return net, endpoints


def gqn(images, poses, scope="GQN"):
  """
  Defines the computational graph of the GQN model.

  Returns:
    net: The last tensor of the network.
    endpoints: A dictionary providing quick access to the most important model
      nodes in the computational graph.
  """
  endpoints = {}

  representation, endpoints_r = tower_encoder(images, poses)

  endpoints.update(endpoints_r)
  endpoints["representation"] = representation

  net = representation
  return net, endpoints

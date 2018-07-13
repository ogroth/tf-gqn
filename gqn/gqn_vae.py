"""
Contains VAE decoders for GQN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .gqn_params import PARAMS
from .gqn_utils import broadcast_pose


def vae_tower_decoder(
    z, query_pose, output_channels=PARAMS.LSTM_CANVAS_CHANNELS,
    scope="VAETowerDecoder"):
  with tf.variable_scope(scope):
    endpoints = {}

    net = z
    net = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    height, width = tf.shape(net)[1], tf.shape(net)[2]
    query_pose = broadcast_pose(query_pose, height, width)
    net = tf.concat([net, query_pose], axis=-1)

    skip1 = tf.layers.conv2d(net, filters=256, kernel_size=1, strides=1,
                             padding="SAME", activation=None)

    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip1

    net = tf.layers.conv2d(net, filters=256, kernel_size=2, strides=2,
                           padding="VALID", activation=tf.nn.relu)

    net = tf.image.resize_bilinear(net, size=(2 * height, 2 * width))

    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    skip2 = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                             padding="SAME", activation=None)
    net = tf.layers.conv2d(net, filters=128, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    # TODO(ogroth): correct implementation for the skip connection?
    net = net + skip2

    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.image.resize_bilinear(net, size=(2 * height, 2 * width))

    net = tf.layers.conv2d(net, filters=128, kernel_size=1, strides=1,
                           padding="SAME", activation=tf.nn.relu)
    net = tf.layers.conv2d(net, filters=256, kernel_size=3, strides=1,
                           padding="SAME", activation=tf.nn.relu)

    net = tf.layers.conv2d(net, filters=output_channels, kernel_size=1,
                           strides=1, padding="SAME", activation=None)

    return net, endpoints

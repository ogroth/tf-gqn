"""
Contains (hyper-)parameters of the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


GQNParams = collections.namedtuple(
    typename='GQNParams',
    field_names=[
        'IMG_HEIGHT',
        'IMG_WIDTH',
        'IMG_CHANNELS',
        'POSE_CHANNELS',
        'LSTM_OUTPUT_CHANNELS',
        'LSTM_CANVAS_CHANNELS',
        'LSTM_KERNEL_SIZE',
        'Z_CHANNELS',
        'REPRESENTATION_CHANNELS',
        'GENERATOR_INPUT_CHANNELS',
        'INFERENCE_INPUT_CHANNELS',
    ])
PARAMS = GQNParams()

# constants
PARAMS.IMG_HEIGHT = 64
PARAMS.IMG_WIDTH = 64
PARAMS.IMG_CHANNELS = 3
PARAMS.POSE_CHANNELS = 7

# hyper-parameters
PARAMS.LSTM_OUTPUT_CHANNELS = 256
PARAMS.LSTM_CANVAS_CHANNELS = 256
PARAMS.LSTM_KERNEL_SIZE = 5

PARAMS.Z_CHANNELS = 64
PARAMS.REPRESENTATION_CHANNELS = 256

PARAMS.GENERATOR_INPUT_CHANNELS = PARAMS.POSE_CHANNELS + PARAMS.REPRESENTATION_CHANNELS + \
                                  PARAMS.Z_CHANNELS

PARAMS.INFERENCE_INPUT_CHANNELS = PARAMS.POSE_CHANNELS + PARAMS.REPRESENTATION_CHANNELS

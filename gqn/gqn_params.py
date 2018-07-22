"""
Contains (hyper-)parameters of the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

_DEFAULTS = {
    # constants
    'IMG_HEIGHT' : 64,
    'IMG_WIDTH' : 64,
    'IMG_CHANNELS' : 3,
    'POSE_CHANNELS' : 7,
    # input parameters
    'CONTEXT_SIZE' : 5,
    # hyper-parameters: scene representation
    'ENC_TYPE' : 'pool',  # encoding architecture used: pool | tower
    'ENC_HEIGHT' : 16,
    'ENC_WIDTH' : 16,
    'ENC_CHANNELS' : 256,
    # hyper-parameters: generator LSTM
    'LSTM_OUTPUT_CHANNELS' : 256,
    'LSTM_CANVAS_CHANNELS' : 256,
    'LSTM_KERNEL_SIZE' : 5,
    'Z_CHANNELS' : 64,  # latent space size per image generation step
    'GENERATOR_INPUT_CHANNELS' : 327,  # pose + representation + z
    'INFERENCE_INPUT_CHANNELS' : 263,  # pose + representation
    'SEQ_LENGTH' : 4,  # number of steps in image generation sequence, orig.: 12
    # hyper-parameters: eta functions
    'ETA_INTERNAL_KERNEL_SIZE' : 5,  # TODO(ogroth): map to internal eta functions
    'ETA_EXTERNAL_KERNEL_SIZE' : 1,  # kernel size for final projection of canvas to mean image
    # hyper-parameters: ADAM optimization
    'ANNEAL_SIGMA_TAU' : 200000,  # annealing interval for global noise
    'GENERATOR_SIGMA_ALPHA' : 2.0,  # start value for global generation variance
    'GENERATOR_SIGMA_BETA' : 0.7,  # final value for global generation variance
    'ANNEAL_LR_TAU' : 1600000,  # annealing interval for learning rate
    'ADAM_LR_ALPHA' : 5 * 10e-4,  # start learning rate of ADAM optimizer, orig.: 5 * 10e-4
    'ADAM_LR_BETA' : 5 * 10e-5,  # final learning rate of ADAM optimizer, orig.: 5 * 10e-5
}

_GQNParams = collections.namedtuple(
    typename='GQNParams',
    field_names=list(_DEFAULTS.keys())
    )
PARAMS = _GQNParams(**_DEFAULTS)

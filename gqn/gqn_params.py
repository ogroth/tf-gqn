"""
Contains (hyper-)parameters of the GQN implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy


GQN_DEFAULT_PARAM_DICT = {
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
    'SEQ_LENGTH' : 8,  # number image generation steps, orig.: 12
    # hyper-parameters: eta functions
    'ETA_INTERNAL_KERNEL_SIZE' : 5,  # internal projection of states to means and variances
    'ETA_EXTERNAL_KERNEL_SIZE' : 1,  # kernel size for final projection of canvas to mean image
    # hyper-parameters: ADAM optimization
    'ANNEAL_SIGMA_TAU' : 200000,  # annealing interval for global noise
    'GENERATOR_SIGMA_ALPHA' : 2.0,  # start value for global generation variance
    'GENERATOR_SIGMA_BETA' : 0.7,  # final value for global generation variance
    'ANNEAL_LR_TAU' : 1600000,  # annealing interval for learning rate
    'ADAM_LR_ALPHA' : 5 * 10e-5,  # start learning rate of ADAM optimizer, orig.: 5 * 10e-4
    'ADAM_LR_BETA' : 5 * 10e-6,  # final learning rate of ADAM optimizer, orig.: 5 * 10e-5
}

GQNConfig = collections.namedtuple(
    typename='GQNConfig',
    field_names=list(GQN_DEFAULT_PARAM_DICT.keys())
    )
GQN_DEFAULT_CONFIG = GQNConfig(**GQN_DEFAULT_PARAM_DICT)


def create_gqn_config(custom_params: dict) -> GQNConfig:
  """
  Customizes the default parameters of the GQN with the parameters specified in
  'custom_params'.

  Returns:
    GQNConfig
  """
  customized_keys = list(
      set(custom_params.keys()).intersection(set(GQN_DEFAULT_PARAM_DICT.keys())))
  customized_params = copy.deepcopy(GQN_DEFAULT_PARAM_DICT)
  for k in customized_keys:
    customized_params[k] = custom_params[k]
  return GQNConfig(**customized_params)

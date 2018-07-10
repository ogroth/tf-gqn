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

from .gqn_params import _GQNParams
from .gqn_encoder import pool_encoder
from .gqn_rnn import inference_rnn, generator_rnn
from .gqn_utils import broadcast_encoding



def gqn(
    query_pose: tf.Tensor, target_frame: tf.Tensor,
    context_poses: tf.Tensor, context_frames: tf.Tensor,
    model_params: _GQNParams, is_training: bool = True,
    scope: str = "GQN"):
  """
  Defines the computational graph of the GQN model.

  Arguments:
    query_pose: Pose vector of the query camera.
    target_frame: Ground truth frame of the query camera.
    context_poses: Camera poses of the context views.
    context_frames: Frames of the context views.
    model_params: Named tuple containing the parameters of the GQN model as \
      defined in gqn_params.py
    is_training: Flag whether graph shall be created in training mode (including \
      the inference module necessary for training the generator). If set to 'False',
      only the generator LSTM will be created.
    scope: Scope name of the graph.

  Returns:
    net: The last tensor of the network.
    endpoints: A dictionary providing quick access to the most important model
      nodes in the computational graph.
  """
  # shorthand notations for model parameters
  _BATCH_SIZE = model_params.BATCH_SIZE
  _CONTEXT_SIZE = model_params.CONTEXT_SIZE
  _DIM_POSE = model_params.POSE_CHANNELS
  _DIM_H_IMG = model_params.IMG_HEIGHT
  _DIM_W_IMG = model_params.IMG_WIDTH
  _DIM_C_IMG = model_params.IMG_CHANNELS
  _DIM_H_ENC = model_params.ENC_HEIGHT
  _DIM_W_ENC = model_params.ENC_WIDTH
  _DIM_C_ENC = model_params.ENC_CHANNELS
  _SEQ_LENGTH = model_params.SEQ_LENGTH

  with tf.variable_scope(scope):
    endpoints = {}

    # pack scene context into pseudo-batch for encoder
    context_poses_packed = tf.reshape(context_poses, shape=[-1, _DIM_POSE])
    context_frames_packed = tf.reshape(
        context_frames, shape=[-1, _DIM_H_IMG, _DIM_W_IMG, _DIM_C_IMG])

    # define scene encoding graph psi
    enc_r_packed, endpoints_psi = pool_encoder(context_frames_packed, context_poses_packed)
    endpoints.update(endpoints_psi)

    # unpack scene encoding and reduce to single vector
    enc_r_packed = tf.reshape(
        enc_r_packed,
        shape=[_BATCH_SIZE, _CONTEXT_SIZE, 1, 1, _DIM_C_ENC])  # 1, 1 for pool encoder only!
    enc_r = tf.reduce_sum(enc_r_packed, axis=1) # add scene representations per data tuple
    endpoints["enc_r"] = enc_r

    # broadcast scene representation to 1/4 of targeted frame size
    enc_r_broadcast = broadcast_encoding(vector=enc_r, height=_DIM_H_ENC, width=_DIM_W_ENC)

    # define generator graph (with inference component if in training mode)
    if is_training:
      mu_target, endpoints_rnn = inference_rnn(
          representations=enc_r_broadcast,
          query_poses=query_pose,
          target_frames=target_frame,
          sequence_size=_SEQ_LENGTH,
      )
    else:
      mu_target, endpoints_rnn = generator_rnn(
          representations=enc_r_broadcast,
          query_poses=query_pose,
          sequence_size=_SEQ_LENGTH
      )

    endpoints.update(endpoints_rnn)
    net = mu_target # final mu tensor parameterizing target frame sampling
    return net, endpoints

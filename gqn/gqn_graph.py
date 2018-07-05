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

from .gqn_params import PARAMS
from .gqn_utils import eta, sample_z


class GeneratorLSTMCell(tf.contrib.rnn.RNNCell):
  # TODO(stefan): better description here
  """GeneratorLSTMCell wrapper that upscales output with a deconvolution and
     adds it to a side input."""

  def __init__(self,
               input_shape,
               output_channels,
               canvas_channels,
               kernel_size=5,
               use_bias=True,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(GeneratorLSTMCell, self).__init__(name=name)

    if len(input_shape) - 1 != 2:
      raise ValueError("Invalid input_shape {}.".format(input_shape))

    self._conv_cell = tf.contrib.rnn.Conv2DLSTMCell(
      input_shape=input_shape,
      output_channels=output_channels,
      kernel_shape=[kernel_size, kernel_size],
      use_bias=use_bias,
      forget_bias=forget_bias,
      initializers=initializers,
      name=name + "_ConvLSTMCell")

    # TODO(stefan,ogroth): do we want to hard-code here the output size of the
    #                      deconvolution to 4?
    canvas_size = tf.TensorShape(
      map(lambda x: 4 * x, input_shape[:-1]) + [canvas_channels])
    self._side_state_channels = canvas_channels
    self._output_size = (canvas_size, self._conv_cell.output_size)
    self._state_size = (canvas_size, self._conv_cell.state_size)

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    """
    :param inputs: concatenated pose, representation, and noise z
    :param state: canvas (u), cell (c), and hidden (h) states
    :param scope:
    :return:
    """
    canvas, (cell_state, hidden_state) = state
    sub_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)

    # run the ConvLSTMCell
    sub_output, new_sub_state = self._conv_cell(inputs, sub_state)

    # upscale the output and add it to u (the side state)
    new_canvas = canvas + tf.layers.conv2d_transpose(
      sub_output, filters=self._side_state_channels, kernel_size=4, strides=4)

    new_output = (new_canvas, sub_output)
    new_state = (new_canvas, new_sub_state)

    return new_output, new_state


def generator_rnn(poses, representations, sequence_size=12,
                  scope="GeneratorRNN"):
  batch = tf.shape(representations)[0]
  height, width = tf.shape(representations)[1], tf.shape(representations)[2]

  cell = GeneratorLSTMCell(
    [height, width, _GENERATOR_INPUT_CHANNELS], _LSTM_OUTPUT_CHANNELS,
    _LSTM_CANVAS_CHANNELS, _LSTM_KERNEL_SIZE, name="GeneratorCell")

  outputs = []
  with tf.variable_scope(scope) as varscope:
    if not tf.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    poses = broadcast_poses(poses, height, width)
    inputs = tf.concat([poses, representations], axis=-1)
    state = cell.zero(batch, tf.float32)

    for time in range(sequence_size):
      if time > 0:
        varscope.reuse_variables()

      z = sample_z(state[1].h, scope="ita_pi")
      (output, state) = cell(tf.concat([inputs, z], axis=-1), state)

      outputs.append(output)

  return outputs[-1][0]


def inference_rnn(poses, representations, sequence_size=12,
                  scope="InferenceRNN"):

  batch = tf.shape(representations)[0]
  height, width = tf.shape(representations)[1], tf.shape(representations)[2]

  # TODO(stefan,ogroth): how are variables shared between inference and
  #                      generator?
  inference_cell = tf.contrib.rnn.Conv2DLSTMCell(
    [height, width, _INFERENCE_INPUT_CHANNELS], _LSTM_OUTPUT_CHANNELS,
    [_LSTM_KERNEL_SIZE, _LSTM_KERNEL_SIZE], name="InferenceCell")

  generator_cell = GeneratorLSTMCell(
    [height, width, _GENERATOR_INPUT_CHANNELS], _LSTM_OUTPUT_CHANNELS,
    _LSTM_CANVAS_CHANNELS, _LSTM_KERNEL_SIZE, name="GeneratorCell")

  outputs = []
  with tf.variable_scope(scope) as varscope:
    if not tf.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    poses = broadcast_poses(poses, height, width)
    inputs = tf.concat([poses, representations], axis=-1)

    inf_state = inference_cell.zero_state(batch, tf.float32)
    gen_state = generator_cell.zero(batch, tf.float32)

    for time in range(sequence_size):
      if time > 0:
        varscope.reuse_variables()

      # TODO(stefan,ogroth): we need x^q, u^l
      inf_input = inputs

      z = sample_z(inf_state.h, scope="ita_q")
      gen_input = tf.concat([inputs, z], axis=-1)

      # TODO(stefan,ogroth): how do you actually use the inference hidden state?
      inf_state[1].h += gen_state[1].h

      (inf_output, inf_state) = inference_cell(inf_input, inf_state)
      (gen_output, gen_state) = generator_cell(gen_input, gen_state)

      outputs.append((inf_output, gen_output))

  return outputs


def gqn(images, poses, scope="GQN"):
  """
  Defines the computational graph of the GQN model.

  Returns:
    net: The last tensor of the network.
    endpoints: A dictionary providing quick access to the most important model
      nodes in the computational graph.
  """
  with tf.variable_scope(scope):
    endpoints = {}
    representation, endpoints_r = tower_encoder(images, poses)

    endpoints.update(endpoints_r)
    endpoints["representation"] = representation

    net = representation
    return net, endpoints

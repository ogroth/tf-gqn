"""
Contains the RNN definition of the DRAW architecture used for image generation.

The DRAW architecture was originally introduced in:
@InProceedings{pmlr-v37-gregor15,
  title = 	 {DRAW: A Recurrent Neural Network For Image Generation},
  author = 	 {Karol Gregor and Ivo Danihelka and Alex Graves and Danilo Rezende and Daan Wierstra},
  booktitle = 	 {Proceedings of the 32nd International Conference on Machine Learning},
  pages = 	 {1462--1471},
  year = 	 {2015},
  editor = 	 {Francis Bach and David Blei},
  volume = 	 {37},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Lille, France},
  month = 	 {07--09 Jul},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v37/gregor15.pdf},
  url = 	 {http://proceedings.mlr.press/v37/gregor15.html},
}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import tensorflow as tf

from .gqn_params import PARAMS
from .gqn_utils import broadcast_pose, create_sub_scope, \
  compute_eta_and_sample_z, sample_z


class GQNLSTMCell(tf.contrib.rnn.RNNCell):
  # TODO(stefan): better description here
  """GeneratorLSTMCell wrapper that upscales output with a deconvolution and
     adds it to a side input."""

  def __init__(self,
               input_shape,
               output_channels,
               kernel_size=5,
               use_bias=True,
               forget_bias=1.0,
               hidden_state_name="h",
               name="GQNCell"):
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
    super(GQNLSTMCell, self).__init__(name=name)

    if len(input_shape) - 1 != 2:
      raise ValueError("Invalid input_shape {}.".format(input_shape))

    # TODO(stefan,ogroth): do we want to hard-code here the output size of the
    #                      deconvolution to 4?
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_size = kernel_size
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._hidden_state_name = hidden_state_name
    state_size = tf.TensorShape(self._input_shape[:-1] + [output_channels])
    self._output_size = state_size
    self._state_size = tf.contrib.rnn.LSTMStateTuple(state_size, state_size)

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    """
    :param inputs: list of inputs
    :param state: cell (c), and hidden (h) states
    :param scope:
    :return:
    """
    cell_state, hidden_state = state

    inputs[self._hidden_state_name] = hidden_state

    with tf.name_scope("InputConv"):
        new_hidden = self._conv(inputs)
        gates = tf.split(value=new_hidden,
                         num_or_size_splits=4,
                         axis=-1)

    input_gate, new_input, forget_gate, output_gate = gates

    with tf.name_scope("Forget"):
      new_cell = tf.nn.sigmoid(forget_gate + self._forget_bias) * cell_state

    with tf.name_scope("Update"):
      new_cell += tf.nn.sigmoid(input_gate) * tf.nn.tanh(new_input)

    with tf.name_scope("Output"):
      output = tf.nn.tanh(new_cell) * tf.nn.sigmoid(output_gate)

    new_state = tf.contrib.rnn.LSTMStateTuple(new_cell, output)

    return output, new_state

  def _conv(self, inputs):
    """
    This is numerically equivalent to concatenating the inputs and then
    convolving everything. However, it allows for splitting the convolution
    variables per input and (with the right scoping) share variables between
    different cells when inputs match.
    """
    conv_outputs = []
    for k in inputs:
      conv_outputs.append(
          tf.layers.conv2d(
              inputs[k],
              filters=4 * self._output_channels,
              kernel_size=self._kernel_size,
              strides=1,
              padding='SAME',
              use_bias=self._use_bias,
              activation=None,
              name="{}_LSTMConv".format(k))
      )

    return tf.add_n(conv_outputs)


_GeneratorCellInput = namedtuple('GeneratorCellInput',
                                 ['representation', 'query_pose', 'z'])
_GeneratorCellOutput = namedtuple('GeneratorCellOutput', ['canvas', 'lstm']) # canvas, h
_GeneratorCellState = namedtuple('GeneratorCellState', ['canvas', 'lstm']) # canvas, (c, h)


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
               name="GeneratorLSTMCell"):
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

    self._gqn_cell = GQNLSTMCell(
        input_shape, output_channels, kernel_size, use_bias, forget_bias,
        hidden_state_name="h_g", name="{}_GQNCell".format(name))

    # TODO(stefan,ogroth): do we want to hard-code here the output size of the
    #                      deconvolution to 4?
    canvas_size = tf.TensorShape([4 * x for x in input_shape[:-1]] + [canvas_channels])
    self._canvas_channels = canvas_channels
    self._output_size = _GeneratorCellOutput(canvas_size,
                                             self._gqn_cell.output_size)
    self._state_size = _GeneratorCellState(canvas_size,
                                           self._gqn_cell.state_size)

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs: _GeneratorCellInput, state: _GeneratorCellState, scope=None) -> _GeneratorCellOutput:
    """
    :param inputs: concatenated pose, representation, and noise z
    :param state: canvas (u), cell (c), and hidden (h) states
    :param scope:
    :return:
    """
    canvas, (cell_state, hidden_state) = state
    sub_state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)

    sub_output, new_sub_state = self._gqn_cell(
      inputs._asdict(), sub_state, scope=create_sub_scope(scope, "GQNCell"))

    # upscale the output and add it to u (the side state)
    new_canvas = canvas + tf.layers.conv2d_transpose(
        sub_output, filters=self._canvas_channels, kernel_size=4, strides=4,
        name="UpsampleGeneratorOutput")

    new_output = _GeneratorCellOutput(new_canvas, sub_output)
    new_state = _GeneratorCellState(new_canvas, new_sub_state)

    return new_output, new_state


_InferenceCellInput = namedtuple('InferenceCellInput',
                                 ['representation', 'query_pose', 'query_image',
                                  'canvas', 'h_g'])
_InferenceCellOutput = namedtuple('InferenceCellOutput', ['lstm']) # h
_InferenceCellState = namedtuple('InferenceCellOutput', ['lstm']) # c, h


class InferenceLSTMCell(tf.contrib.rnn.RNNCell):
  # TODO(stefan): better description here
  """InferenceLSTMCell wrapper that upscales output with a deconvolution and
     adds it to a side input."""

  def __init__(self,
               input_shape,
               output_channels,
               kernel_size=5,
               use_bias=True,
               forget_bias=1.0,
               name="InferenceLSTMCell"):
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
    super(InferenceLSTMCell, self).__init__(name=name)

    if len(input_shape) - 1 != 2:
      raise ValueError("Invalid input_shape {}.".format(input_shape))

    self._gqn_cell = GQNLSTMCell(
        input_shape, output_channels, kernel_size, use_bias, forget_bias,
        hidden_state_name='h_e', name="{}_GQNCell".format(name))

    # TODO(stefan,ogroth): do we want to hard-code here the output size of the
    #                      deconvolution to 4?
    self._output_size = _InferenceCellOutput(self._gqn_cell.output_size)
    self._state_size = _InferenceCellState(self._gqn_cell.state_size)

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs: _InferenceCellInput, state: _InferenceCellState, scope=None) -> _InferenceCellOutput:
    """
    :param inputs: concatenated pose, representation, and noise z
    :param state:
    :param scope:
    :return:
    """
    cell_state, hidden_state = state.lstm

    input_dict = inputs._asdict()
    query_image = input_dict.pop('query_image')
    canvas = input_dict.pop('canvas')

    # TODO(stefan,ogroth): actually, this could just be another input, what
    #                      is the right thing to do?
    # downscale the query image and canvas to add it to the hidden state
    input_canvas_and_image = tf.layers.conv2d(
        tf.concat([query_image, canvas], axis=-1),
        filters=self.output_size.lstm[-1], kernel_size=4, strides=4,
        padding='VALID', use_bias=False,
        name="DownsampleInferenceInputCanvasAndImage")
    hidden_state += input_canvas_and_image

    state = tf.contrib.rnn.LSTMStateTuple(cell_state, hidden_state)
    output, new_state = self._gqn_cell(
      input_dict, state, scope=create_sub_scope(scope, "GQNCell"))

    return _InferenceCellOutput(output), _InferenceCellState(new_state)


def generator_rnn(representations, query_poses, sequence_size=12,
                  scope="GQN_RNN"):

  dim_r = representations.get_shape().as_list()
  batch = tf.shape(representations)[0]
  height, width = dim_r[1], dim_r[2]

  cell = GeneratorLSTMCell(
      input_shape=[height, width, PARAMS.GENERATOR_INPUT_CHANNELS],
      output_channels=PARAMS.LSTM_OUTPUT_CHANNELS,
      canvas_channels=PARAMS.LSTM_CANVAS_CHANNELS,
      kernel_size=PARAMS.LSTM_KERNEL_SIZE,
      name="GeneratorCell")

  outputs = []
  endpoints = {}
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as varscope:
    if not tf.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    query_poses = broadcast_pose(query_poses, height, width)
    state = cell.zero_state(batch, tf.float32)

    # unroll generator LSTM
    for step in range(sequence_size):
      z = sample_z(state.lstm.h, scope="Sample_eta_pi")
      inputs = _GeneratorCellInput(representations, query_poses, z)
      with tf.name_scope("Generator"):
        (output, state) = cell(inputs, state, "LSTM")

      # register enpoints
      ep_canvas = "canvas_%d" % (step, )
      endpoints[ep_canvas] = output.canvas

      # aggregate outputs
      outputs.append(output)

    # compute final mu tensor parameterizing sampling of target frame
    target_canvas = outputs[-1].canvas
    mu_target, _, _ = compute_eta_and_sample_z(target_canvas, scope="Sample_eta_g")
    endpoints['mu_target'] = mu_target

  return mu_target, endpoints


def inference_rnn(representations, query_poses, target_frames, sequence_size=12,
                  scope="GQN_RNN"):
  """
  TODO(ogroth): write docstring!
  """

  dim_r = representations.get_shape().as_list()
  batch = tf.shape(representations)[0]
  height, width = dim_r[1], dim_r[2]

  generator_cell = GeneratorLSTMCell(
      input_shape=[height, width, PARAMS.GENERATOR_INPUT_CHANNELS],
      output_channels=PARAMS.LSTM_OUTPUT_CHANNELS,
      canvas_channels=PARAMS.LSTM_CANVAS_CHANNELS,
      kernel_size=PARAMS.LSTM_KERNEL_SIZE,
      name="GeneratorCell")
  inference_cell = InferenceLSTMCell(
      input_shape=[height, width, PARAMS.INFERENCE_INPUT_CHANNELS],
      output_channels=PARAMS.LSTM_OUTPUT_CHANNELS,
      kernel_size=PARAMS.LSTM_KERNEL_SIZE,
      name="InferenceCell")

  outputs = []
  endpoints = {}
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as varscope:
    if not tf.executing_eagerly():
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    query_poses = broadcast_pose(query_poses, height, width)

    inf_state = inference_cell.zero_state(batch, tf.float32)
    gen_state = generator_cell.zero_state(batch, tf.float32)

    # unroll the LSTM cells
    for step in range(sequence_size):

      # TODO(stefan,ogroth): What is the correct order for sampling, inference
      # and generator update?
      # 1) sample; 2) infer; 3) generate
      inf_input = _InferenceCellInput(
          representations, query_poses, target_frames, gen_state.canvas,
          gen_state.lstm.h)
      mu_q, sigma_q, z_q = compute_eta_and_sample_z(inf_state.lstm.h,
                                                    scope="Sample_eta_q")
      mu_pi, sigma_pi, z_pi = compute_eta_and_sample_z(gen_state.lstm.h,
                                                       scope="Sample_eta_pi")
      gen_input = _GeneratorCellInput(representations, query_poses, z_q)

      # generator and inference cell need to have the same variable scope
      # for variable sharing!
      with tf.name_scope("Inference"):
        (inf_output, inf_state) = inference_cell(inf_input, inf_state, "LSTM")
      with tf.name_scope("Generator"):
        (gen_output, gen_state) = generator_cell(gen_input, gen_state, "LSTM")

      # register enpoints
      ep_mu_q = "mu_q_%d" % (step, )
      ep_mu_pi = "mu_pi_%d" % (step, )
      ep_sigma_q = "sigma_q_%d" % (step, )
      ep_sigma_pi = "sigma_pi_%d" % (step, )
      endpoints[ep_mu_q] = mu_q
      endpoints[ep_mu_pi] = mu_pi
      endpoints[ep_sigma_q] = sigma_q
      endpoints[ep_sigma_pi] = sigma_pi

      # aggregate outputs
      outputs.append((inf_output, gen_output))

    # compute final mu tensor parameterizing sampling of target frame
    target_canvas = outputs[-1][1].canvas
    mu_target, _, _ = compute_eta_and_sample_z(
        target_canvas, channels=PARAMS.IMG_CHANNELS, scope="Sample_eta_g")
    endpoints['mu_target'] = mu_target

  return mu_target, endpoints

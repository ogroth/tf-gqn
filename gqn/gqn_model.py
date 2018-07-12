"""
Contains the tf.estimator compatible model definition for GQN.

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

from .gqn_graph import gqn
from .gqn_params import _GQNParams


def _linear_annealing_scheme(gqn_params: _GQNParams):
  """
  Defines the computational graph for the global sigma annealing scheme used in
  image sampling.
  """
  anneal_alpha = tf.constant(gqn_params.GENERATOR_SIGMA_ALPHA, dtype=tf.float32)
  anneal_beta = tf.constant(gqn_params.GENERATOR_SIGMA_BETA, dtype=tf.float32)
  anneal_tau = tf.constant(gqn_params.GENERATOR_SIGMA_TAU, dtype=tf.float32)
  anneal_step = tf.minimum(tf.train.get_global_step(), gqn_params.GENERATOR_SIGMA_TAU)
  anneal_diff =  anneal_alpha - anneal_beta
  anneal_coeff = anneal_step / anneal_tau
  sigma_target = anneal_alpha - anneal_coeff * anneal_diff
  return sigma_target


def gqn_model_fn(features, labels, mode, params):
  """
  Defines an tf.estimator.EstimatorSpec for the GQN model.

  Args:
    features: Query = collections.namedtuple('Query', ['context', 'query_camera'])
    labels: tf.Tensor of the target image
    mode:
    params:
      gqn_params: _GQNParams type containing the model parameters
      debug: bool; if true, model will produce additional debug output
        tensorboard summaries for image generation process 
  
  Returns:
    spec: tf.estimator.EstimatorSpec
  """
  # shorthand notations for parameters
  _SEQ_LENGTH = params['gqn_params'].SEQ_LENGTH

  # feature and label mapping according to gqn_input_fn
  query_pose = features.query.query_camera
  target_frame = labels
  context_poses = features.query.context.cameras
  context_frames = features.query.context.frames

  # graph setup
  net, ep_gqn = gqn(
      query_pose=query_pose,
      target_frame=target_frame,
      context_poses=context_poses,
      context_frames=context_frames,
      model_params=params['gqn_params'],
      is_training=(mode == tf.estimator.ModeKeys.TRAIN)
  )

  # collect intermediate endpoints
  mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
  for i in range(_SEQ_LENGTH):
    mu_q.append(ep_gqn["mu_q_%d" % i])
    sigma_q.append(ep_gqn["sigma_q_%d" % i])
    mu_pi.append(ep_gqn["mu_pi_%d" % i])
    sigma_pi.append(ep_gqn["sigma_pi_%d" % i])

  # outputs: sampled images
  mu_target = net
  sigma_target = _linear_annealing_scheme(params['gqn_params'])
  target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
  target_sample = target_normal.sample()
  # image generation steps in debug mode
  # TODO(ogroth): add sampling from intermediate endpoints

  # predictions
  predictions = {
      'target_sample' : target_sample
  }

  # eval_metric_ops

  # ELBO setup
  # train & eval summary hooks
  # optimizer
  # train_op

  raise NotImplementedError()

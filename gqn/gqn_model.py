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

from .gqn_graph import gqn_draw, gqn_vae
from .gqn_objective import gqn_draw_elbo, gqn_vae_elbo
from .gqn_params import _GQNParams
from .gqn_utils import debug_canvas_image_mean


def _linear_noise_annealing(gqn_params: _GQNParams) -> tf.Tensor:
  """
  Defines the computational graph for the global sigma annealing scheme used in
  image sampling.
  """
  sigma_i = tf.constant(gqn_params.GENERATOR_SIGMA_ALPHA, dtype=tf.float32)
  sigma_f = tf.constant(gqn_params.GENERATOR_SIGMA_BETA, dtype=tf.float32)
  step = tf.cast(tf.train.get_global_step(), dtype=tf.float32)
  tau = tf.constant(gqn_params.ANNEAL_SIGMA_TAU, dtype=tf.float32)
  sigma_target = tf.maximum(
      sigma_f + (sigma_i - sigma_f) * (1.0 - step / tau),
      sigma_f)
  return sigma_target

def _linear_lr_annealing(gqn_params: _GQNParams) -> tf.Tensor:
  """
  Defines the computational graph for the global learning rate annealing scheme
  used during optimization.
  """
  eta_i = tf.constant(gqn_params.ADAM_LR_ALPHA, dtype=tf.float32)
  eta_f = tf.constant(gqn_params.ADAM_LR_BETA, dtype=tf.float32)
  step = tf.cast(tf.train.get_global_step(), dtype=tf.float32)
  tau = tf.constant(gqn_params.ANNEAL_LR_TAU, dtype=tf.float32)
  lr = tf.maximum(
      eta_f + (eta_i - eta_f) * (1.0 - step / tau),
      eta_f)
  return lr


def gqn_draw_model_fn(features, labels, mode, params):
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
  _CONTEXT_SIZE = params['gqn_params'].CONTEXT_SIZE
  _SEQ_LENGTH = params['gqn_params'].SEQ_LENGTH

  # feature and label mapping according to gqn_input_fn
  query_pose = features.query_camera
  target_frame = labels
  context_poses = features.context.cameras
  context_frames = features.context.frames

  # graph setup
  net, ep_gqn = gqn_draw(
      query_pose=query_pose,
      target_frame=target_frame,
      context_poses=context_poses,
      context_frames=context_frames,
      model_params=params['gqn_params'],
      is_training=(mode != tf.estimator.ModeKeys.PREDICT)
  )

  # outputs: sampled images
  mu_target = net
  sigma_target = _linear_noise_annealing(params['gqn_params'])
  target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
  target_sample = tf.identity(target_normal.sample(), name='target_sample')
  # write out image summaries in debug mode
  if params['debug']:
    for i in range(_CONTEXT_SIZE):
      tf.summary.image(
          'context_frame_%d' % (i + 1),
          context_frames[:, i],
          max_outputs=1
      )
    tf.summary.image(
        'target_images',
        labels,
        max_outputs=1
    )
    tf.summary.image(
        'target_means',
        mu_target,
        max_outputs=1
    )
    generator_sequence = debug_canvas_image_mean(
        [ep_gqn['canvas_{}'.format(i)] for i in range(_SEQ_LENGTH)]
    )
    tf.summary.image(
        'generator_sequence_mean',
        generator_sequence,
        max_outputs=1
    )

  # predictions to make when deployed during test time
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'target_sample' : target_sample
    }

  # ELBO setup
  if mode != tf.estimator.ModeKeys.PREDICT:
    # collect intermediate endpoints
    mu_q, sigma_q, mu_pi, sigma_pi = [], [], [], []
    for i in range(_SEQ_LENGTH):
      mu_q.append(ep_gqn["mu_q_%d" % i])
      sigma_q.append(ep_gqn["sigma_q_%d" % i])
      mu_pi.append(ep_gqn["mu_pi_%d" % i])
      sigma_pi.append(ep_gqn["sigma_pi_%d" % i])
    elbo, ep_elbo = gqn_draw_elbo(
        mu_target, sigma_target,
        mu_q, sigma_q,
        mu_pi, sigma_pi,
        target_frame)
    if params['debug']:
      tf.summary.scalar(
          name='target_llh',
          tensor=ep_elbo['target_llh']
      )
      tf.summary.scalar(
          name='kl_regularizer',
          tensor=ep_elbo['kl_regularizer']
      )

  # optimization
  if mode == tf.estimator.ModeKeys.TRAIN:
    lr = _linear_lr_annealing(params['gqn_params'])
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(
        loss=elbo,
        global_step=tf.train.get_global_step()
    )

  # evaluation metrics to monitor
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'mean_abs_pixel_error' : tf.metrics.mean_absolute_error(
            labels=target_frame,
            predictions=mu_target)
    }

  # create SpecSheet
  if mode == tf.estimator.ModeKeys.TRAIN:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=elbo,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=elbo,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  return estimator_spec


def gqn_vae_model_fn(features, labels, mode, params):
  """
  Defines an tf.estimator.EstimatorSpec for the GQN-VAE baseline model.

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

  # feature and label mapping according to gqn_input_fn
  query_pose = features.query_camera
  target_frame = labels
  context_poses = features.context.cameras
  context_frames = features.context.frames

  # graph setup
  net, ep_gqn_vae = gqn_vae(
      query_pose=query_pose,
      context_poses=context_poses,
      context_frames=context_frames,
      model_params=params['gqn_params'],
  )

  # collect intermediate endpoints
  mu_q, sigma_q = ep_gqn_vae["mu_q"], ep_gqn_vae["sigma_q"]

  # outputs: sampled images
  mu_target = net
  sigma_target = _linear_noise_annealing(params['gqn_params'])
  target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
  target_sample = tf.identity(target_normal.sample(), name='target_sample')
  # write out image summaries in debug mode
  if params['debug']:
    tf.summary.image(
        'target_images',
        labels,
        max_outputs=1
    )
    tf.summary.image(
        'target_sample',
        target_sample,
        max_outputs=1
    )

  # predictions to make when deployed during test time
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'target_sample' : target_sample
    }

  # ELBO setup
  if mode != tf.estimator.ModeKeys.PREDICT:
    elbo = gqn_vae_elbo(
        mu_target, sigma_target,
        mu_q, sigma_q,
        target_frame)

  # optimization
  if mode == tf.estimator.ModeKeys.TRAIN:
    # TODO(ogroth): tune hyper-parameters of optimizer?
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=elbo,
        global_step=tf.train.get_global_step()
    )

  # evaluation metrics to monitor
  if mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'mean_abs_pixel_error' : tf.metrics.mean_absolute_error(
            labels=target_frame,
            predictions=target_sample)
    }

  # create SpecSheet
  if mode == tf.estimator.ModeKeys.TRAIN:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=elbo,
        train_op=train_op)
  if mode == tf.estimator.ModeKeys.EVAL:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=elbo,
        eval_metric_ops=eval_metric_ops)
  if mode == tf.estimator.ModeKeys.PREDICT:
    estimator_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  return estimator_spec

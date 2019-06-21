"""
Contains the definition of ELBO objective used to train the GQN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List
import tensorflow as tf


# ---------- ad-hoc data types ----------

TfTensorList = List[tf.Tensor]


# ---------- public APIs for objective graph definition ----------

def gqn_draw_elbo(
    mu_target: tf.Tensor, sigma_target: tf.Tensor,
    mu_q: TfTensorList, sigma_q: TfTensorList,
    mu_pi: TfTensorList, sigma_pi: TfTensorList,
    target_frame: tf.Tensor,
    scope='GQN_DRAW_ELBO'):
  """
  Defines the ELBO objective of the GQN graph.

  Arguments:
    mu_target: The mean parameterizing the final image sampling.
    sigma_target: The sigma parameterizing the final image sampling.
    mu_q: A list of mus parameterizing the posterior for every image generation step.
    sigma_q: A list of sigmas parameterizing the posterior for every image generation step.
    mu_pi: A list of mus parameterizing the prior for every image generation step.
    sigma_pi: A list of sigmas parameterizing the prior for every image generation step.
    target_frame: The ground truth target frame to produce (i.e. the 'label').
    scope: The variable scope name of the objective graph.

  Returns:
    elbo: Scalar. Expected value over the negative log-likelihood of the target frame given \
      the target distribution regularized by the cumulative KL divergence between posterior \
      and prior distributions at every image generation step.
    endpoints: A dictionary of relevant computational endpoints for quick access to the graph \
      nodes.
  """
  with tf.variable_scope(scope):
    endpoints = {}
    # negative log-likelihood of target frame given target distribution
    target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
    target_llh = tf.identity(
        input=-tf.reduce_sum(
            tf.reduce_mean(target_normal.log_prob(target_frame), axis=0)),
        name='target_llh')
    endpoints['target_llh'] = target_llh
    # KL divergence regularizer over all generation steps
    kl_div_list = []
    for mu_q_l, sigma_q_l, mu_pi_l, sigma_pi_l in zip(mu_q, sigma_q, mu_pi, sigma_pi):
      posterior_normal_l = tf.distributions.Normal(loc=mu_q_l, scale=sigma_q_l)
      prior_normal_l = tf.distributions.Normal(loc=mu_pi_l, scale=sigma_pi_l)
      kl_div_l = tf.distributions.kl_divergence(posterior_normal_l, prior_normal_l)
      # kl_div_l = tf.Print(
      #     input_=kl_div_l,
      #     data=[tf.reduce_sum(tf.cast(tf.is_nan(kl_div_l), tf.float32))]
      # )  # debug
      kl_div_list.append(kl_div_l)
    kl_regularizer = tf.identity(
        input=tf.reduce_sum(
            tf.reduce_mean(tf.add_n(kl_div_list), axis=0)),
        name='kl_regularizer')
    endpoints['kl_regularizer'] = kl_regularizer
    # final ELBO term
    # target_llh = tf.Print(input_=target_llh, data=[target_llh])  # debug
    # kl_regularizer = tf.Print(input_=kl_regularizer, data=[kl_regularizer])  # debug
    elbo = target_llh + kl_regularizer
    return elbo, endpoints


def gqn_vae_elbo(
    mu_target: tf.Tensor, sigma_target: tf.Tensor,
    mu_q: tf.Tensor, sigma_q: tf.Tensor,
    target_frame: tf.Tensor,
    scope='GQN_VAE_ELBO'):
  """
  [WIP] This GQN version is currently not maintained!
  
  Defines the ELBO of the GQN-VAE baseline graph.

  Arguments:
    mu_target: The mean parameterizing the final image sampling.
    sigma_target: The sigma parameterizing the final image sampling.
    mu_q: The mean parameterizing the posterior for image generation.
    sigma_q: The sigma parameterizing the posterior for image generation.
    target_frame: The ground truth target frame to produce (i.e. the 'label').
    scope: The variable scope name of the objective graph.

  Returns:
    elbo: Scalar. Expected value over the negative log-likelihood of the target frame given \
      the target distribution regularized by the cumulative KL divergence between posterior \
      and prior distributions for image generation.
  """
  with tf.variable_scope(scope):
    # negative log-likelihood of target frame given target distribution
    target_normal = tf.distributions.Normal(loc=mu_target, scale=sigma_target)
    target_llh = tf.identity(
        input=-tf.reduce_sum(
            tf.reduce_mean(target_normal.log_prob(target_frame), axis=0)),
        name='target_llh')

    # KL divergence regularizer
    posterior_normal = tf.distributions.Normal(loc=mu_q, scale=sigma_q)
    prior_normal = tf.distributions.Normal(loc=tf.zeros_like(mu_q),
                                          scale=tf.ones_like(sigma_q))
    kl_div = tf.distributions.kl_divergence(posterior_normal, prior_normal)
    kl_regularizer = tf.identity(
        input=tf.reduce_sum(
            tf.reduce_mean(tf.add_n(kl_div), axis=0)),
        name='kl_regularizer')

    # final ELBO term
    elbo = target_llh + kl_regularizer
    return elbo

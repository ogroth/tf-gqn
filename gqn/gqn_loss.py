"""
Contains the definition of ELBO objective used to train the GQN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def gqn_elbo():
  """
  Defines the ELBO of the GQN graph.

  Arguments:
    mu_target: The mean parameterizing the final image sampling.
    sigma_target: The sigma parameterizing the final image sampling.
    mu_q: A list of mus parameterizing the posterior for every image generation step.
    sigma_q: A list of sigmas parameterizing the posterior for every image generation step.
    mu_pi: A list of mus parameterizing the prior for every image generation step.
    sigma_pi: A list of sigmas parameterizing the prior for every image generation step.
    target_frame: The ground truth target frame to produce (i.e. the 'label').
  
  Returns:
    elbo: Scalar. Expected value over the negative log-likelihood of the target frame given \
      the target distribution regularized by the cumulative KL divergence between posterior \
      and prior distributions at every image generation step.
  """
  pass

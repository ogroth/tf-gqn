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

# HYPERPARAMETER CONSTANTS
# STUB

def gqn_model_fn(features, labels, mode, params) -> tf.estimator.EstimatorSpec:
  """
  Defines an tf.estimator.EstimatorSpec for the GQN model.

  Args:
    features:
    labels:
    mode:
    params:
  
  Returns:
    spec: tf.estimator.EstimatorSpec
  """
  #TODO: implement
  raise NotImplementedError()

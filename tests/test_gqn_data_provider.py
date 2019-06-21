"""
Test script to check the data input pipeline.
"""

import os
import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_ROOT = os.path.abspath(os.path.join(SCRIPT_PATH, '..'))
sys.path.append(TF_GQN_ROOT)

import tensorflow as tf
import numpy as np

from data_provider.gqn_provider import gqn_input_fn


# constants
DATASET_ROOT_PATH = os.path.join(TF_GQN_ROOT, 'data', 'gqn-dataset')
DATASET_NAME = 'rooms_ring_camera'
CTX_SIZE = 5 # number of context (image, pose) pairs for a given query pose
BATCH_SIZE = 36

# graph definition
dataset = gqn_input_fn(
    dataset_name=DATASET_NAME,
    root=DATASET_ROOT_PATH,
    mode=tf.estimator.ModeKeys.TRAIN,
    context_size=CTX_SIZE,
    batch_size=BATCH_SIZE)
iterator = dataset.make_initializable_iterator()
data = iterator.get_next()

# fetch one batch of data
with tf.train.SingularMonitoredSession() as sess:
  sess.run(iterator.initializer)
  features, labels = sess.run(data)
  # print shapes of fetched objects
  print("Shapes of fetched tensors:")
  print("Query camera poses: %s" % str(features.query.query_camera.shape))
  print("Target images: %s" % str(features.target.shape))
  print("Context camera poses: %s" % str(features.query.context.cameras.shape))
  print("Context frames: %s" % str(features.query.context.frames.shape))

print("TEST PASSED!")

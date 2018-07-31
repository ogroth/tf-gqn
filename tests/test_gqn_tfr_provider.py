"""
Quick test script to check the data reader.
"""

import os
import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, '..'))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import numpy as np

from data_provider.gqn_tfr_provider import DataReader

# TODO(ogroth): make CLI parameters!
ROOT_PATH = '/tmp/data/gqn-dataset'
DATASET_NAME = 'rooms_ring_camera'
CTX_SIZE = 5 # number of context (image, pose) pairs for a given query pose
BATCH_SIZE = 36

# graph definition
data_reader = DataReader(dataset=DATASET_NAME, context_size=CTX_SIZE, root=ROOT_PATH)
data = data_reader.read(batch_size=BATCH_SIZE)

# fetch one batch of data
with tf.train.SingularMonitoredSession() as sess:
  d = sess.run(data)
  # print shapes of fetched objects
  print("Shapes of fetched tensors:")
  print("Query camera poses: %s" % str(d.query.query_camera.shape))
  print("Target images: %s" % str(d.target.shape))
  print("Context camera poses: %s" % str(d.query.context.cameras.shape))
  print("Context frames: %s" % str(d.query.context.frames.shape))

print("TEST PASSED!")

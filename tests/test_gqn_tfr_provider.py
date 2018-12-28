"""
Quick test script to check the data reader.
"""

import os
import sys
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
TF_GQN_HOME = os.path.abspath(os.path.join(SCRIPT_PATH, '..'))
sys.path.append(TF_GQN_HOME)

import tensorflow as tf
import argparse

from data_provider.gqn_tfr_provider import DataReader

parser = argparse.ArgumentParser(description='Test the DataReader')
parser.add_argument(
    '--root_dir', type=str, default='/tmp/data/gqn-dataset',
    help='The path to the gqn-dataset directory.'
)
parser.add_argument(
    '--dataset', type=str, default='rooms_ring_camera',
    help='The name of the GQN dataset to use. \
    Available names are: \
    jaco | mazes | rooms_free_camera_no_object_rotations | \
    rooms_free_camera_with_object_rotations | rooms_ring_camera | \
    shepard_metzler_5_parts | shepard_metzler_7_parts'
)
parser.add_argument(
    '--context_size', type=int, default=5,
    help='Number of context (image, pose) pairs for a given query pose'
)
parser.add_argument(
    '--batch_size', type=int, default=36,
    help='Batch size'
)
FLAGS = parser.parse_args()


# graph definition
data_reader = DataReader(dataset=FLAGS.dataset, context_size=FLAGS.context_size, root=FLAGS.root_dir)
data = data_reader.read(batch_size=FLAGS.batch_size)

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

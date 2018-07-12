"""
Training script to train GQN as a tf.estimator.Estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import tensorflow as tf

from gqn.gqn_model import gqn_model_fn
from data_provider.gqn_tfr_provider import gqn_input_fn
from gqn.gqn_params import PARAMS

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train a GQN as a tf.estimator.Estimator.')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/Volumes/Maxtor/datasets/gqn-dataset',
    help='The path to the gqn-dataset directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='rooms_ring_camera',
    help='The name of the GQN dataset to use. \
    Available names are: \
    jaco | mazes | rooms_free_camera_no_object_rotations | \
    rooms_free_camera_with_object_rotations | rooms_ring_camera | \
    shepard_metzler_5_parts | shepard_metzler_7_parts')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/gqn',
    help='The directory where the model will be stored.')
# training parameters
ARGPARSER.add_argument(
    '--train_epochs', type=int, default=40,
    help='The number of epochs to train.')
# memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=12,
    help='The number of data points per batch. One data point is a tuple of \
    ((query_camera_pose, [(context_frame, context_camera_pose)]), target_frame).')
ARGPARSER.add_argument(
    '--memcap', type=float, default=0.8,
    help='Maximum fraction of memory to allocate per GPU.')
# data loading
ARGPARSER.add_argument(
    '--queue_threads', type=int, default=4,
    help='How many parallel threads to run for data queuing.')
ARGPARSER.add_argument(
    '--queue_buffer', type=int, default=256,
    help='How many batches to queue up.')


def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """
  # using the Winograd non-fused algorithms provides a small performance boost
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # set up a RunConfig and the estimator
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=FLAGS.memcap
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
      save_checkpoints_secs=1e9 #TODO: make parameter
  )
  model_params = {
      'gqn_params' : PARAMS,
  }
  classifier = tf.estimator.Estimator(
      model_fn=gqn_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params=model_params,
  )

  # main loop
  for _ in range(FLAGS.train_epochs):

    # create logging hooks
    # tensors_to_log = {
    #     'logits' : 'logits'
    # }
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=1000) #TODO: make parameter

    # train the model for one epoch
    train_input = lambda: gqn_input_fn(
        dataset=FLAGS.dataset,
        context_size=PARAMS.CONTEXT_SIZE,
        root=FLAGS.data_dir,
        mode=tf.estimator.ModeKeys.TRAIN,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.train_epochs,
        num_threads=FLAGS.queue_threads,
        buffer_size=FLAGS.queue_buffer,
    )
    classifier.train(
        input_fn=train_input,
        # hooks=[logging_hook],
    )

    # evaluate the model on the validation set
    eval_input = lambda: gqn_input_fn(
        dataset=FLAGS.dataset,
        context_size=PARAMS.CONTEXT_SIZE,
        root=FLAGS.data_dir,
        mode=tf.estimator.ModeKeys.EVAL,
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.train_epochs,
        num_threads=FLAGS.queue_threads,
        buffer_size=FLAGS.queue_buffer,
    )
    eval_results = classifier.evaluate(
        input_fn=eval_input,
    )


if __name__ == '__main__':
  print("Training a GQN.")
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGV)

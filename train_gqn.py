"""
Training script to train GQN as a tf.estimator.Estimator.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import json

import tensorflow as tf

from data_provider.gqn_provider import gqn_input_fn
from gqn.gqn_model import gqn_draw_model_fn
from gqn.gqn_params import create_gqn_config
from utils.runscript import save_run_command


# ---------- command line arguments ----------
ARGPARSER = argparse.ArgumentParser(
    description='Train a GQN as a tf.estimator.Estimator.')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='data/gqn-dataset',
    help='The path to the gqn-dataset directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='rooms_ring_camera',
    help='The name of the GQN dataset to use. \
      Available names are: \
      jaco | mazes | rooms_free_camera_no_object_rotations | \
      rooms_free_camera_with_object_rotations | rooms_ring_camera | \
      shepard_metzler_5_parts | shepard_metzler_7_parts')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='models/gqn',
    help='The directory where the model will be stored.')
# model parameters
ARGPARSER.add_argument(
    '--seq_length', type=int, default=8,
    help='The number of generation steps of the DRAW LSTM.')
ARGPARSER.add_argument(
    '--context_size', type=int, default=5,
    help='The number of context points.')
ARGPARSER.add_argument(
    '--img_size', type=int, default=64,
    help='Height and width of the squared input images.')
ARGPARSER.add_argument(
    '--enc_type', type=str, default='pool',
    help='The encoding architecture type.')
# solver parameters
ARGPARSER.add_argument(
    '--adam_lr_alpha', type=float, default=5*10e-5,
    help='The initial learning rate of the ADAM solver.')
ARGPARSER.add_argument(
    '--adam_lr_beta', type=float, default=5*10e-6,
    help='The final learning rate of the ADAM solver.')
ARGPARSER.add_argument(
    '--anneal_lr_tau', type=int, default=1600000,
    help='The interval over which to anneal the learning rate from lr_alpha to \
      lr_beta.')
# training parameters
ARGPARSER.add_argument(
    '--train_epochs', type=int, default=2,
    help='The number of epochs to train.')
# snapshot parameters
ARGPARSER.add_argument(
    '--chkpt_steps', type=int, default=10000,
    help='Number of steps between checkpoint saves.')
# memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=36,
    help='The number of data points per batch.')
ARGPARSER.add_argument(
    '--memcap', type=float, default=1.0,
    help='Maximum fraction of memory to allocate per GPU.')
# data loading
ARGPARSER.add_argument(
    '--queue_threads', type=int, default=4,
    help='How many parallel threads to run for data queuing.')
ARGPARSER.add_argument(
    '--queue_buffer', type=int, default=4,
    help='How many batches to queue up.')
# logging
ARGPARSER.add_argument(
    '--log_steps', type=int, default=100,
    help='Global steps between log output.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")
ARGPARSER.add_argument(
    '--initial_eval', default=False, action='store_true',
    help="Runs an evaluation before the first training iteration.")


# ---------- helper functions ----------

def _save_model_config(config: dict, run_dir, name):
  """
  Saves a model config (dict) into the run directory in JSON format.
  """
  fn = "%s.json" % (name, )
  with open(os.path.join(run_dir, fn), 'w') as f:
    json.dump(config, f, indent=2, sort_keys=True)

def _load_model_config(run_dir, name):
  """
  Loads a model config from a JSON file and returns a config dict.
  """
  fn = "%s.json" % (name, )
  with open(os.path.join(run_dir, fn), 'r') as f:
    config = json.load(f)
  return config


# ---------- main ----------

def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """
  # using the Winograd non-fused algorithms provides a small performance boost
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # set up the model directory
  os.makedirs(name=ARGS.model_dir, exist_ok=True)

  # save run cmd
  save_run_command(argparser=ARGPARSER, run_dir=ARGS.model_dir)

  # set up a RunConfig and the estimator
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=ARGS.memcap
  )
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
      save_checkpoints_steps=ARGS.chkpt_steps,
  )

  # set up the model config
  config_name = "gqn_config"
  config_fn = "%s.json" % (config_name, )
  config_path = os.path.join(ARGS.model_dir, config_fn)
  if os.path.exists(config_path):  # load model config from previous run
    custom_params = _load_model_config(ARGS.model_dir, config_name)
    gqn_config = create_gqn_config(custom_params)
    print("Loaded existing model config from %s" % (config_path, ))
  else:  # create new model config from CLI parameters
    custom_params = {
        'IMG_HEIGHT' : ARGS.img_size,
        'IMG_WIDTH' : ARGS.img_size,
        'CONTEXT_SIZE' : ARGS.context_size,
        'SEQ_LENGTH' : ARGS.seq_length,
        'ENC_TYPE' : ARGS.enc_type,
        'ENC_HEIGHT' : ARGS.img_size // 4,  # must be 1/4 of target frame height
        'ENC_WIDTH' : ARGS.img_size // 4,  # must be 1/4 of target frame width
        'ADAM_LR_ALPHA' : ARGS.adam_lr_alpha,
        'ADAM_LR_BETA' : ARGS.adam_lr_beta,
        'ANNEAL_LR_TAU' : ARGS.anneal_lr_tau,
    }
    gqn_config = create_gqn_config(custom_params)
    _save_model_config(gqn_config._asdict(), ARGS.model_dir, config_name)  # save config to restore later
    print("Saved model config to %s" % (config_path, ))
  model_params = {
      'gqn_params' : gqn_config,
      'model_dir' : ARGS.model_dir,
      'debug' : ARGS.debug,
  }

  # set up tf.estimator
  model = tf.estimator.Estimator(
      model_fn=gqn_draw_model_fn,
      model_dir=ARGS.model_dir,
      config=run_config,
      params=model_params,
  )

  # set up input_fn (data pipeline)
  input_fn = lambda estimator_mode: gqn_input_fn(
      dataset_name=ARGS.dataset,
      root=ARGS.data_dir,
      mode=estimator_mode,
      context_size=gqn_config.CONTEXT_SIZE,
      batch_size=ARGS.batch_size,
      custom_frame_size=ARGS.img_size,
      num_threads=ARGS.queue_threads,
      buffer_size=ARGS.queue_buffer,
  )
  train_input = lambda: input_fn(estimator_mode=tf.estimator.ModeKeys.TRAIN)
  eval_input = lambda: input_fn(estimator_mode=tf.estimator.ModeKeys.EVAL)

  # create logging hooks
  tensors_to_log = {
      'l2_reconstruction' : 'l2_reconstruction'
  }
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=ARGS.log_steps
  )

  # optional initial evaluation
  if ARGS.initial_eval:
    eval_results = model.evaluate(
        input_fn=eval_input,
        hooks=[logging_hook],
    )

  # main loop
  for _ in range(ARGS.train_epochs):
    model.train(
        input_fn=train_input,
        hooks=[logging_hook],
    )
    eval_results = model.evaluate(
        input_fn=eval_input,
        hooks=[logging_hook],
    )


if __name__ == '__main__':
  print("Training a GQN.")
  ARGS, UNPARSED_ARGS = ARGPARSER.parse_known_args()
  print("PARSED ARGV:", ARGS)
  print("UNPARSED ARGV:", UNPARSED_ARGS)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGS)

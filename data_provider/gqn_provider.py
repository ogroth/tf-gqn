"""
Input pipeline (tf.dataset and input_fn) for GQN datasets.
Adapted from the implementation provided here:
https://github.com/deepmind/gqn-datasets/blob/acca9db6d9aa7cfa4c41ded45ccb96fecc9b272e/data_reader.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf


# ---------- ad-hoc data structures ----------

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


# ---------- dataset constants ----------

_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    # super-small subset of rooms_ring for debugging purposes
    rooms_ring_camera_debug=DatasetInfo(
        basepath='rooms_ring_camera_debug',
        train_size=1,
        test_size=1,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


# ---------- helper functions ----------

def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)

def _get_dataset_files(dataset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dataset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dataset_info.train_size
  else:
    num_files = dataset_info.test_size
  length = len(str(num_files))
  template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
  record_paths = [  # indexing runs from 1 to n
      os.path.join(base, template.format(i, num_files))
      for i in range(1, num_files + 1)]
  return record_paths

def _get_randomized_indices(context_size, dataset_info, seed):
  """Generates randomized indices into a sequence of a specific length."""
  example_size = context_size + 1
  indices = tf.range(0, dataset_info.sequence_size)
  indices = tf.random_shuffle(indices, seed=seed)
  indices = tf.slice(indices, begin=[0], size=[example_size])
  return indices

def _parse(raw_data, dataset_info):
  """Parses raw data from the tfrecord."""
  feature_map = {
      'frames': tf.FixedLenFeature(
          shape=dataset_info.sequence_size, dtype=tf.string),
      'cameras': tf.FixedLenFeature(
          shape=[dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
          dtype=tf.float32)
  }
  # example = tf.parse_example(raw_data, feature_map)
  example = tf.parse_single_example(raw_data, feature_map)
  return example

def _preprocess(example, indices, context_size, custom_frame_size, dataset_info):
  """Preprocesses the parsed data."""
  # frames
  example_size = context_size + 1
  frames = tf.concat(example['frames'], axis=0)
  frames = tf.gather(frames, indices, axis=0)
  frames = tf.map_fn(
      _convert_frame_data, tf.reshape(frames, [-1]),
      dtype=tf.float32, back_prop=False)
  dataset_image_dimensions = tuple(
      [dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
  frames = tf.reshape(
      frames, (example_size, ) + dataset_image_dimensions)
  if (custom_frame_size and
      custom_frame_size != dataset_info.frame_size):
    frames = tf.reshape(frames, dataset_image_dimensions)
    new_frame_dimensions = (custom_frame_size,) * 2 + (_NUM_CHANNELS,)
    frames = tf.image.resize_bilinear(
        frames, new_frame_dimensions[:2], align_corners=True)
    frames = tf.reshape(
        frames, (-1, example_size) + new_frame_dimensions)
  # cameras
  raw_pose_params = example['cameras']
  raw_pose_params = tf.reshape(
      raw_pose_params,
      [dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
  raw_pose_params = tf.gather(raw_pose_params, indices, axis=0)
  pos = raw_pose_params[:, 0:3]
  yaw = raw_pose_params[:, 3:4]
  pitch = raw_pose_params[:, 4:5]
  cameras = tf.concat(
      [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=-1)
  # return preprocessed tuple
  preprocessed_example = {}
  preprocessed_example['frames'] = frames
  preprocessed_example['cameras'] = cameras
  return preprocessed_example

def _prepare(preprocessed_example):
  """Prepares the preprocessed data into (feature, label) tuples."""
  # decompose
  frames = preprocessed_example['frames']
  cameras = preprocessed_example['cameras']
  # split data
  context_frames = frames[:-1]
  context_cameras = cameras[:-1]
  target = frames[-1]
  query_camera = cameras[-1]
  context = Context(cameras=context_cameras, frames=context_frames)
  query = Query(context=context, query_camera=query_camera)
  data = TaskData(query=query, target=target)
  return data, data.target

# ---------- input_fn ----------

def gqn_input_fn(
    dataset_name,
    root,
    mode,
    context_size,
    batch_size=1,
    num_epochs=1,
    # optionally reshape frames
    custom_frame_size=None,
    # queue params
    num_threads=4,
    buffer_size=256,
    seed=None):
  """
  Creates a tf.data.Dataset based op that returns data.
    Args:
      dataset_name: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
          'rooms_free_camera_no_object_rotations',
          'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
          'shepard_metzler_7_parts'].
      root: string, path to the root folder of the data.
      mode: one of tf.estimator.ModeKeys.
      context_size: integer, number of views to be used to assemble the context.
      batch_size: (optional) batch size, defaults to 1.
      num_epochs: (optional) number of times to go through the dataset,
          defaults to 1.
      custom_frame_size: (optional) integer, required size of the returned
          frames, defaults to None.
      num_threads: (optional) integer, number of threads used to read and parse
          the record files, defaults to 4.
      buffer_size: (optional) integer, capacity of the underlying prefetch or
          shuffle buffer, defaults to 256.
      seed: (optional) integer, seed for the random number generators used in
          the dataset.

    Returns:
      tf.data.dataset yielding tuples of the form (features, labels)
      shapes:
        features.query.context.cameras: [N, K, 7]
        features.query.context.frames: [N, K, H, W, 3]
        features.query.query_camera: [N, 7]
        features.target (same as labels): [N, H, W, 3]

    Raises:
      ValueError: if the required version does not exist; if the required mode
         is not supported; if the requested context_size is bigger than the
         maximum supported for the given dataset version.
  """

  # map estimator mode key to dataset internal mode strings
  if mode == tf.estimator.ModeKeys.TRAIN:
    str_mode = 'train'
  else:
    str_mode = 'test'
  # check validity of requested dataset and split
  if dataset_name not in _DATASETS:
    raise ValueError('Unrecognized dataset {} requested. Available datasets '
                      'are {}'.format(dataset_name, _DATASETS.keys()))
  if str_mode not in _MODES:
    raise ValueError('Unsupported mode {} requested. Supported modes '
                      'are {}'.format(str_mode, _MODES))
  # retrieve dataset parameters
  dataset_info = _DATASETS[dataset_name]
  if context_size >= dataset_info.sequence_size:
    raise ValueError(
        'Maximum support context size for dataset {} is {}, but '
        'was {}.'.format(
            dataset_name, dataset_info.sequence_size-1, context_size))
  # collect the paths to all tfrecord files
  record_paths = _get_dataset_files(dataset_info, str_mode, root)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(
      filenames=record_paths, num_parallel_reads=num_threads)
  # parse the data from tfrecords
  dataset = dataset.map(
      lambda raw_data: _parse(raw_data, dataset_info),
      num_parallel_calls=num_threads)
  # preprocess into context and target
  indices = _get_randomized_indices(context_size, dataset_info, seed)
  dataset = dataset.map(
      lambda example: _preprocess(example, indices, context_size, custom_frame_size, dataset_info),
      num_parallel_calls=num_threads)
  # parse into tuple expected by tf.estimator input_fn
  dataset = dataset.map(_prepare, num_parallel_calls=num_threads)
  # shuffle data
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(buffer_size=(buffer_size * batch_size), seed=seed)
  # set up batching
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size)
  return dataset

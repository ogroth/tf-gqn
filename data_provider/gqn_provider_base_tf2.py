"""
Input pipeline (tf.dataset and input_fn) for GQN datasets.
Adapted from the implementation provided here:
https://github.com/deepmind/gqn-datasets/blob/acca9db6d9aa7cfa4c41ded45ccb96fecc9b272e/data_reader.py

Minimal data reader for GQN TFRecord datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf

from tensorflow.python.util import nest

DatasetInfo = collections.namedtuple(
    "DatasetInfo",
    ["basepath", "train_size", "test_size", "frame_size", "sequence_size"],
)
Context = collections.namedtuple("Context", ["frames", "cameras"])
Query = collections.namedtuple("Query", ["context", "query_camera"])
TaskData = collections.namedtuple("TaskData", ["query", "target"])

_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ("train", "test")

_DATASETS = dict(
    jaco=DatasetInfo(
        basepath="jaco", train_size=3600, test_size=400, frame_size=64, sequence_size=11
    ),
    mazes=DatasetInfo(
        basepath="mazes",
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300,
    ),
    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_with_object_rotations",
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10,
    ),
    rooms_ring_camera=DatasetInfo(
        basepath="rooms_ring_camera",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10,
    ),
    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath="rooms_free_camera_no_object_rotations",
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10,
    ),
    shepard_metzler_5_parts=DatasetInfo(
        basepath="shepard_metzler_5_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15,
    ),
    shepard_metzler_7_parts=DatasetInfo(
        basepath="shepard_metzler_7_parts",
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15,
    ),
)


def _get_dataset_files(dateset_info, mode, rootdir):
    """Generates lists of files for a given dataset version."""
    basepath = dateset_info.basepath
    base = os.path.join(rootdir, basepath, mode)

    return [os.path.join(base, f) for f in os.listdir(base)]


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class EagerDataReader(object):
    """Minimal eager TFRecord reader for Tensorflow >2.0b.

    You can use this reader to load the datasets used to train Generative Query
    Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
    See README.md for a description of the datasets and an example of how to use
    the reader.
    """

    def __init__(
        self,
        dataset,
        context_size,
        rootdir,
        mode="train",
        batch_size=1,
        num_epochs=1,
        # Optionally reshape frames
        custom_frame_size=None,
        # Optionally control dataset object
        num_threads=4,
        buffer_size=256,
        seed=None,
    ):

        if dataset not in _DATASETS:
            raise ValueError(
                "Unrecognized dataset {} requested. Available datasets "
                "are {}".format(dataset, _DATASETS.keys())
            )

        # Dataset description
        self._dataset_info = _DATASETS[dataset]

        if mode not in _MODES:
            raise ValueError(
                "Unsupported mode {} requested. Supported modes are {}".format(
                    mode, _MODES
                )
            )

        if context_size >= self._dataset_info.sequence_size:
            raise ValueError(
                "Maximum support context size for dataset {} is {}, but was {}.".format(
                    dataset, self._dataset_info.sequence_size - 1, context_size
                )
            )

        self.seed = seed
        self._context_size = context_size
        # Number of views in the context + target view
        self._example_size = context_size + 1
        self._custom_frame_size = custom_frame_size

        self.indices = self._get_randomized_indices()

        filenames = _get_dataset_files(self._dataset_info, mode, rootdir)

        # Load the dataset from files
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_threads)
        dataset = dataset.shuffle(buffer_size=(buffer_size * batch_size), seed=seed)
        dataset = dataset.map(self._parse_record, num_parallel_calls=num_threads)
        dataset = dataset.map(self._preprocess, num_parallel_calls=num_threads)
        dataset = dataset.map(self._prepare, num_parallel_calls=num_threads)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)

        self.dataset = dataset

    def _prepare(self, frames, cameras):
        """Reads batch_size (query, target) pairs."""
        context_frames = frames[:, :-1]
        context_cameras = cameras[:, :-1]
        target = frames[:, -1]
        query_camera = cameras[:, -1]
        context = Context(cameras=context_cameras, frames=context_frames)
        query = Query(context=context, query_camera=query_camera)

        return TaskData(query=query, target=target)

    def _parse_record(self, record):
        feature_map = {
            "frames": tf.io.FixedLenFeature(
                shape=self._dataset_info.sequence_size, dtype=tf.string
            ),
            "cameras": tf.io.FixedLenFeature(
                shape=[self._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
                dtype=tf.float32,
            ),
        }
        example = tf.io.parse_single_example(record, feature_map)

        return example

    def _preprocess(self, example):
        """Preprocess raw frames and cameras from parsed tfrecord"""

        frames = self._preprocess_frames(example, self.indices)
        cameras = self._preprocess_cameras(example, self.indices)

        return frames, cameras

    def _get_randomized_indices(self):
        """Generates randomized indices into a sequence of a specific length."""

        indices = tf.range(0, self._dataset_info.sequence_size)
        indices = tf.random.shuffle(indices, seed=self.seed)
        indices = tf.slice(indices, begin=[0], size=[self._example_size])
        return indices

    def _preprocess_frames(self, example, indices):
        """Instantiates the ops used to preprocess the frames data."""
        frames = tf.concat(example["frames"], axis=0)
        frames = tf.gather(frames, indices, axis=0)
        frames = tf.map_fn(
            _convert_frame_data,
            tf.reshape(frames, [-1]),
            dtype=tf.float32,
            back_prop=False,
        )
        dataset_image_dimensions = tuple(
            [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS]
        )
        frames = tf.reshape(frames, (-1, self._example_size) + dataset_image_dimensions)
        if (
            self._custom_frame_size
            and self._custom_frame_size != self._dataset_info.frame_size
        ):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(
                frames, new_frame_dimensions[:2], method=tf.image.ResizeMethod.BILINEAR
            )
            frames = tf.reshape(frames, (-1, self._example_size) + new_frame_dimensions)
        return frames

    def _preprocess_cameras(self, example, indices):
        """Instantiates the ops used to preprocess the cameras data."""
        raw_pose_params = example["cameras"]
        raw_pose_params = tf.reshape(
            raw_pose_params,
            [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS],
        )
        raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2
        )

        return cameras

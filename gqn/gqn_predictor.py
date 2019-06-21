"""
Contains a canned predictor for a GQN.
"""

import os
import json

import numpy as np
import tensorflow as tf

from .gqn_graph import gqn_draw
from .gqn_params import create_gqn_config


def _normalize_pose(pose):
  """
  Converts a camera pose into the GQN format.
  Args:
    pose: [x, y, z, yaw, pitch]; x, y, z in [-1, 1]; yaw, pitch in euler degree
  Returns:
    [x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch)]
  """
  norm_pose = np.zeros((7, ))
  norm_pose[0:3] = pose[0:3]
  norm_pose[3] = np.cos(np.deg2rad(pose[3]))
  norm_pose[4] = np.sin(np.deg2rad(pose[3]))
  norm_pose[5] = np.cos(np.deg2rad(pose[4]))
  norm_pose[6] = np.sin(np.deg2rad(pose[4]))
  # print("Normalized pose: %s -> %s" % (pose, norm_pose))  # DEBUG
  return norm_pose


class GqnViewPredictor(object):
  """
  GQN-based view predictor.
  """

  def __init__(self, model_dir):
    """
    Instantiates a GqnViewPredictor from a saved checkpoint.

    Args:
      model_dir: Path to a GQN model. Must contain 'gqn_config.json', 'checkpoint'
        and 'model.ckpt-nnnnnn'.

    Returns:
      GqnViewPredictor
    """
    # load gqn_config.json
    with open(os.path.join(model_dir, 'gqn_config.json'), 'r') as f:
      gqn_config_dict = json.load(f)
    self._cfg = create_gqn_config(gqn_config_dict)
    self._ctx_size = self._cfg.CONTEXT_SIZE
    self._dim_pose = self._cfg.POSE_CHANNELS
    self._dim_img_h = self._cfg.IMG_HEIGHT
    self._dim_img_w = self._cfg.IMG_WIDTH
    self._dim_img_c = self._cfg.IMG_CHANNELS
    # create input placeholders
    self._ph_ctx_poses = tf.placeholder(
        shape=[1, self._ctx_size, self._dim_pose],
        dtype=tf.float32)
    self._ph_ctx_frames = tf.placeholder(
        shape=[1, self._ctx_size, self._dim_img_h, self._dim_img_w, self._dim_img_c],
        dtype=tf.float32)
    self._ph_query_pose = tf.placeholder(
        shape=[1, self._dim_pose], dtype=tf.float32)
    self._ph_tgt_frame = tf.placeholder(  # just used for graph construction
        shape=[1, self._dim_img_h, self._dim_img_w, self._dim_img_c],
        dtype=tf.float32)
    # re-create gqn graph
    self._net, self._ep = gqn_draw(
        query_pose=self._ph_query_pose,
        context_frames=self._ph_ctx_frames,
        context_poses=self._ph_ctx_poses,
        target_frame=self._ph_tgt_frame,
        model_params=self._cfg,
        is_training=False)
    print(">>> Instantiated GQN:")  # DEBUG
    for name, ep in self._ep.items():
      print("\t%s\t%s" % (name, ep))
    # create session
    self._sess = tf.InteractiveSession()
    # load snapshot
    saver = tf.train.Saver()
    ckpt_path = tf.train.latest_checkpoint(model_dir)
    saver.restore(self._sess, save_path=ckpt_path)
    print(">>> Restored parameters from: %s" % (ckpt_path, ))  # DEBUG
    # create data placeholders
    self._context_frames = []  # list of RGB frames [H, W, C]
    self._context_poses = []  # list of normalized poses [x, y, z, cos(yaw), sin(yaw), cos(pitch), sin(pitch)]

  @property
  def sess(self):
    """Expose the underlying tensorflow session."""
    return self._sess

  @property
  def frame_resolution(self):
    """Returns the video resolution as (H, W, C)."""
    return (self._dim_img_h, self._dim_img_w, self._dim_img_c)

  def add_context_view(self, frame: np.ndarray, pose: np.ndarray):
    """
    Adds a (frame, pose) tuple as context point for view interpolation.
    Args:
      frame: [H, W, C], in [0, 1]
      pose: [x, y, z, yaw, pitch]; x, y, z in [-1, 1]; yaw, pitch in euler degree
    """
    assert (frame >= 0.0).all() and (frame <= 1.0).all(), \
      "The context frame is not normalized in [0.0, 1.0] (float)."
    assert frame.shape == self.frame_resolution, \
      "The context frame's shape %s does not fit the model's shape %s." % \
      (frame.shape, self.frame_resolution)
    assert pose.shape == (self._dim_pose, ) or pose.shape == (5, ), \
      "The pose's shape %s does not match the specification (either %s or %s)." % \
      (pose.shape, self._dim_pose, (5, ))
    if pose.shape == (5, ):  # assume un-normalized pose
      pose = _normalize_pose(pose)
    # add frame-pose pair to context
    self._context_frames.append(frame)
    self._context_poses.append(pose)

  def clear_context(self):
    """Clears the current context."""
    self._context_frames.clear()
    self._context_poses.clear()

  def render_query_view(self, pose: np.ndarray):
    """
    Renders the scene from the given camera pose.
    Args:
      pose: [x, y, z, yaw, pitch]; x, y, z in [-1, 1]; yaw, pitch in euler degree
    """
    assert len(self._context_frames) >= self._ctx_size \
      and len(self._context_poses) >= self._ctx_size, \
      "Not enough context points available. Required %d. Given: %d" % \
      (self._ctx_size, np.min(len(self._context_frames), len(self._context_poses)))
    assert pose.shape == (self._dim_pose, ) or pose.shape == (5, ), \
      "The pose's shape %s does not match the specification (either %s or %s)." % \
      (pose.shape, self._dim_pose, (5, ))
    if pose.shape == (5, ):  # assume un-normalized pose
      pose = _normalize_pose(pose)
    ctx_frames = np.expand_dims(
        np.stack(self._context_frames[-self._ctx_size:]), axis=0)
    ctx_poses = np.expand_dims(
        np.stack(self._context_poses[-self._ctx_size:]), axis=0)
    query_pose = np.expand_dims(pose, axis=0)
    feed_dict = {
        self._ph_query_pose : query_pose,
        self._ph_ctx_frames : ctx_frames,
        self._ph_ctx_poses : ctx_poses
    }
    [pred_frame] = self._sess.run([self._net], feed_dict=feed_dict)
    pred_frame = np.clip(pred_frame, a_min=0.0, a_max=1.0)
    return pred_frame

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class definition for Tensorboard wrapper."""

import os
import tensorflow as tf
import torch


class TBManager:
  """Simple wrapper for Tensorboard."""

  def __init__(self, exp_dir, task_names, to_track, splits=['train', 'valid']):
    # Set up summaries
    summaries = {}
    placeholders = {}

    with tf.device('/cpu:0'):
      for task in task_names:
        summaries[task] = {}

        for s in splits:
          tmp_summaries = []

          for k in to_track:
            if k not in placeholders:
              placeholders[k] = tf.placeholder(tf.float32, [])
            tmp_summaries += [
                tf.summary.scalar('%s_%s_%s' % (task, s, k), placeholders[k])
            ]

          summaries[task][s] = tf.summary.merge(tmp_summaries)

      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'  # Suppress TF warnings

      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)

      self.writer = tf.summary.FileWriter(exp_dir, self.sess.graph)
      self.summaries = summaries
      self.placeholders = placeholders

  def update(self, task, split, step, vals):
    """Write an update to events file.

    Args:
      task: Index of task to write summary for
      split: Specify 'train' or 'valid'
      step: Current training iteration
      vals: Dictionary with values to update
    """

    # Get summary update for Tensorboard
    feed_dict = {
        self.placeholders[k]: v.cpu() if isinstance(v, torch.Tensor) else v
        for k, v in vals.items()
    }
    summary = self.sess.run(self.summaries[task][split], feed_dict=feed_dict)

    # Log data
    self.writer.add_summary(summary, step)
    self.writer.flush()

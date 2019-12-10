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

from mtl.train import multiclass
import numpy as np
import torch


def setup_extra_args(parser):
  parser.add_argument('--dist_loss_wt', type=float, default=1)
  parser.add_argument('--class_loss_wt', type=float, default=0)


class Task(multiclass.Task):
  """Distillation training manager."""

  def __init__(self, opt, ds, dataloaders):
    super().__init__(opt, ds, dataloaders)
    self.ce_loss = torch.nn.CrossEntropyLoss()
    self.mse_loss = torch.nn.MSELoss()

  def run(self, split, step):
    opt, ds = self.opt, self.ds
    self.step = step
    self.split = split

    # Sample task
    task_idx = self.sample_task()
    self.task_idx = task_idx
    self.curr_task = ds['train'][task_idx].task_name

    # Get samples + model output
    inp, label, _ = self.get_next_sample(split, task_idx)
    ref_feats, pred_feats, pred = self.model(inp, task_idx, split, step)

    # Calculate loss
    _, class_preds = torch.max(pred, 1)
    t_min, t_max = self.model.task_low[task_idx], self.model.task_high[task_idx]
    accuracy = class_preds.eq(label).float().mean()
    accuracy = (accuracy - t_min) / (t_max - t_min)

    class_loss = self.ce_loss(pred, label)
    distill_loss = self.mse_loss(pred_feats, ref_feats.detach())

    self.net_loss = 0
    if opt.dist_loss_wt:
      self.net_loss += opt.dist_loss_wt * distill_loss
    if opt.class_loss_wt:
      self.net_loss += opt.class_loss_wt * class_loss

    if split == 'valid':
      self.valid_accuracy_track[task_idx] += [accuracy.data.item()]
    self.update_log('accuracy', accuracy.data.item())
    self.update_log('network_loss', self.net_loss.data.item())
    self.score = np.array([d['valid'] for d in self.log['accuracy']]).mean()

    self.global_trained_steps += 1
    self.task_trained_steps[task_idx] += 1

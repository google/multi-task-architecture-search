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

from mtl.meta.optim import optimizer
from mtl.util import tensorboard_manager
import numpy as np
import torch


def setup_extra_args(parser):
  parser.add_argument('--optimizer', type=str, default='SGD')
  parser.add_argument('-l', '--learning_rate', type=float, default=1)
  parser.add_argument('--momentum', type=float, default=.5)
  parser.add_argument('--init_random', type=int, default=32)
  parser.add_argument('--num_deltas', type=int, default=16)
  parser.add_argument('--num_to_use', type=int, default=8)
  parser.add_argument('--delta_size', type=float, default=.05)
  parser.add_argument('--do_weight_reg', type=int, default=0)
  parser.add_argument('--diag_weight_decay', type=float, default=.01)
  parser.add_argument('--multiobjective', type=int, default=0)


class MetaOptimizer(optimizer.MetaOptimizer):

  def __init__(self, opt):
    self.num_tasks = len(opt.task_choice.split('-'))
    super().__init__(opt)

    self.ref_params = self.curr_params._parameters[opt.search[0]]
    self.task_grad_masks = np.array([
        self.ref_params.get_task_parameter_mask(i)
        for i in range(self.num_tasks)
    ])

  def tensorboard_setup(self):
    self.task_names = ['total'] + ['task_%d' % i for i in range(self.num_tasks)]
    self.to_track = ['score', 'param_use']
    self.tb = tensorboard_manager.TBManager(self.exp_dir, self.task_names,
                                            self.to_track, ['train'])

  def run(self, cmd_queue, result_queue):
    opt = self.opt

    self.base_cmd = self.cfg.base_cmd

    if '--' in opt.unparsed:
      # Additional arguments to pass on to experiments
      extra_args = opt.unparsed[opt.unparsed.index('--') + 1:]
      self.extra_child_args = extra_args

    if self.best_exp is None:
      # Sample a random set of parameters and keep best
      print('Testing initial random samples...')

      samples = []
      for i in range(opt.init_random):
        samples += [self.init_sample()]
        self.submit_cmd(cmd_queue, str(i), param=samples[i])

      _, scores = self.collect_batch_results(opt.init_random, result_queue)
      self.best_score = scores.max()
      self.best_exp = scores.argmax()
      self.best_params = samples[self.best_exp]

    # Set up optimizer
    self.curr_params = self.best_params
    curr_params = self.curr_params.get_params(opt.search)
    curr_params.requires_grad = True
    self.setup_optimizer([curr_params])

    # Save checkpoint
    self.score = self.best_score
    self.save(self.exp_dir + '/snapshot')

    # Main loop
    while self.exp_count < opt.num_samples:
      # Sample deltas
      deltas = []
      p1 = curr_params.data.clone()
      for delta_idx in range(opt.num_deltas):
        p2 = self.ref_params.mutate(p1, delta=opt.delta_size)
        deltas += [np.array(p2 - p1)]
      deltas = np.stack(deltas)

      to_test = np.concatenate([deltas, -deltas], 0)
      samples, exp_ref = [], []
      n_samples = to_test.shape[0]

      for delta_idx in range(n_samples):
        exp_id = str(delta_idx)
        sample = self.init_sample()
        params = curr_params.clone()
        params += torch.Tensor(to_test[delta_idx]).view(params.shape)
        params.data = params.data.clamp(*self.ref_params.valid_range)
        sample.update_params(opt.search, params)
        samples += [sample]
        exp_ref += [self.exp_count]

        self.submit_cmd(cmd_queue, exp_id, param=sample)

      # Collect results
      result_idx_ref, scores = self.collect_batch_results(
          n_samples, result_queue)
      self.score = scores.mean()
      print(self.exp_count, '%.3f' % self.score)

      # Calculate per-task scores
      task_scores = np.zeros((scores.shape[0], self.num_tasks))
      for sample_idx, result_idx in enumerate(result_idx_ref):
        acc = self.results[result_idx][1]['log']['accuracy']
        task_scores[sample_idx] = [acc_['valid'] for acc_ in acc]

      # Update tensorboard
      param_use = np.array(
          curr_params.view(self.ref_params.shape).data[0][0].diag())
      param_use = [param_use.mean()] + list(param_use)
      tmp_scores = [self.score] + list(task_scores.mean(0))
      for task_idx, task_name in enumerate(self.task_names):
        self.tb.update(task_name, 'train', self.exp_count, {
            'score': tmp_scores[task_idx],
            'param_use': param_use[task_idx]
        })

      if scores.max() > self.best_score:
        self.best_score = scores.max()
        self.best_params = samples[scores.argmax()]
        self.best_exp = exp_ref[scores.argmax()]

      if not opt.multiobjective:
        # Single objective optimization

        # Normalize scores
        scores /= scores.std() + 1e-4
        scores = np.stack([scores[:opt.num_deltas], scores[opt.num_deltas:]], 1)

        # Get best deltas
        max_rewards = np.max(scores, axis=1)
        best_idxs = np.argsort(-max_rewards)[:opt.num_to_use]
        rewards = scores[best_idxs]
        tmp_deltas = np.array(deltas)[best_idxs]

        # Calculate weighted sum
        reward_diff = rewards[:, 0] - rewards[:, 1]
        result = -np.dot(reward_diff, tmp_deltas) / reward_diff.size

      else:
        # Multi-objective optimization

        # Normalize scores
        task_scores /= task_scores.std(0, keepdims=True) + 1e-4
        task_scores = np.stack(
            [task_scores[:opt.num_deltas], task_scores[opt.num_deltas:]],
            2).transpose(1, 0, 2)

        # Per-task optimization
        task_results = []

        for task_idx in range(self.num_tasks):
          tmp_scores = task_scores[task_idx]

          # Get best deltas
          max_rewards = np.max(tmp_scores, axis=1)
          best_idxs = np.argsort(-max_rewards)[:opt.num_to_use]
          rewards = tmp_scores[best_idxs]
          tmp_deltas = np.array(deltas)[best_idxs]

          # Calculate weighted sum
          reward_diff = rewards[:, 0] - rewards[:, 1]
          result = -np.dot(reward_diff, tmp_deltas) / reward_diff.size
          result *= self.task_grad_masks[task_idx]
          task_results += [result]

        result = np.stack(task_results, 0).sum(0)
        result /= np.maximum(self.task_grad_masks.sum(0), 1)

      result = torch.Tensor(result)

      # Calculate regularization (only for diagonal terms in forward matrix)
      lmda_mat = torch.unsqueeze(
          torch.stack([
              torch.eye(self.num_tasks),
              torch.zeros(self.num_tasks, self.num_tasks)
          ]), 0)
      lmda_mat = lmda_mat.repeat(opt.num_unique, 1, 1, 1).view(-1)
      lmda_mat *= opt.diag_weight_decay
      if opt.do_weight_reg == 1:
        # Do L1 regularization
        result += lmda_mat * torch.sign(curr_params.data.view(lmda_mat.shape))
      elif opt.do_weight_reg == 2:
        # Do L2 regularization
        result += lmda_mat * curr_params.data.view(lmda_mat.shape)

      curr_params.grad = result.view(curr_params.shape)

      # Update parameters
      self.optim.step()
      curr_params.data = curr_params.data.clamp(*self.ref_params.valid_range)
      self.curr_params.update_params(opt.search, curr_params)

      # Save checkpoint
      self.save(self.exp_dir + '/snapshot')

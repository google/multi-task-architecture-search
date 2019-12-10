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

import importlib

from mtl.meta.optim import optimizer
import numpy as np


class MetaOptimizer(optimizer.MetaOptimizer):

  def __init__(self, opt):
    self.cfg = importlib.import_module('mtl.config.' + opt.config)
    self.num_exps = len(self.cfg.exp_list)
    self.exps_done = np.zeros((opt.num_samples, self.num_exps))
    self.exp_scores = np.zeros_like(self.exps_done)
    super().__init__(opt)

  def checkpoint_ref_setup(self):
    super().checkpoint_ref_setup()
    self.checkpoint_ref['extra'] = ['score', 'exps_done', 'exp_scores']

  def run(self, cmd_queue, result_queue):
    opt = self.opt
    self.base_cmd = self.cfg.base_cmd

    # Set up any additional arguments to tack on to experiments
    extra_args = []
    unparsed = opt.unparsed
    if '--' in unparsed:
      extra_args = unparsed[unparsed.index('--') + 1:]
    self.extra_child_args = extra_args

    print('Number of experiments:', self.num_exps)
    print('Number of trials:', opt.num_samples)
    print('Extra args:', extra_args)

    # Submit all experiments that haven't been run
    for trial_idx in range(opt.num_samples):
      for exp_idx, e in enumerate(self.cfg.exp_list):
        if not self.exps_done[trial_idx, exp_idx]:
          exp_id = 'trial_%d/%s' % (trial_idx, self.cfg.exp_names[exp_idx])
          if '--' in e:
            extra_child_args = e[e.index('--'):]
            e = e[:e.index('--')]
          else:
            extra_child_args = []

          self.submit_cmd(
              cmd_queue,
              exp_id,
              extra_args=e,
              extra_child_args=extra_child_args)

    # Collect results
    while self.exps_done.sum() != opt.num_samples * self.num_exps:
      result = result_queue.get()
      self.results += [result]

      exp_id = result[0].split('/')
      for tmp_idx, val in enumerate(exp_id):
        if 'trial_' in val:
          trial_idx = int(val.split('_')[-1])
          ref_idx = tmp_idx + 1
      exp_idx = self.cfg.exp_names.index('/'.join(exp_id[ref_idx:]))

      score = result[1]['score']
      self.exp_scores[trial_idx, exp_idx] = score
      self.exps_done[trial_idx, exp_idx] = 1

      if score > self.best_score:
        self.best_score = score
        self.best_exp = result[0]

      self.exp_count += 1

      print('Collected %s with score %.2f' % (result[0], score))

      if opt.num_samples < 100 or self.exp_count % 20 == 0:
        self.save(self.exp_dir + '/snapshot')

    self.save(self.exp_dir + '/snapshot')

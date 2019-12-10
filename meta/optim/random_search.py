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

import numpy as np
import subprocess
import torch
import argparse

from mtl.meta.optim import optimizer


class MetaOptimizer(optimizer.MetaOptimizer):

  def __init__(self, opt):
    self.exps_done = None
    self.exp_scores = None
    super().__init__(opt)

  def checkpoint_ref_setup(self):
    super().checkpoint_ref_setup()
    self.checkpoint_ref['extra'] = ['score', 'exps_done', 'exp_scores']

  def setup_samples(self):
    all_samples = []
    for i in range(self.opt.num_samples):
      all_samples += [self.init_sample()]
    return all_samples

  def run(self, cmd_queue, result_queue):
    opt = self.opt
    self.base_cmd = self.cfg.base_cmd

    samples = self.setup_samples()
    n_exps = len(samples)
    print(n_exps, 'parameterizations to test')

    for i, s in enumerate(samples):
      self.submit_cmd(cmd_queue, str(i), param=s)

    self.exp_scores = np.zeros(n_exps)
    self.exps_done = np.zeros(n_exps)

    for i in range(n_exps):
      result = result_queue.get()
      self.results += [result]
      sample_id = int(result[0].split('/')[-1])
      self.exps_done[sample_id] = 1
      if result[1] is not None:
        self.exp_scores[sample_id] = result[1]['score']
      else:
        self.exp_scores[sample_id] = self.worst_score

      self.best_score = self.exp_scores.max()
      self.best_exp = self.exp_scores.argmax()

      if n_exps < 100 or i % 50 == 0:
        self.save(self.exp_dir + '/snapshot')

    self.save(self.exp_dir + '/snapshot')

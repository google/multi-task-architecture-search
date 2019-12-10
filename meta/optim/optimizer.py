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

""" Base metaoptimizer class """
import importlib
import os
import subprocess

from mtl.config.opts.parser import mkdir_p
from mtl.util import session_manager
import numpy as np
import torch


class MetaOptimizer(session_manager.SessionManager):

  def __init__(self, opt, ds=None, dataloaders=None):
    super().__init__(opt)

    self.cfg = importlib.import_module('mtl.config.' + opt.config)
    self.optim = None
    self.curr_params = None
    self.results = []
    self.exp_count = 0
    self.worst_score = opt.worst_score
    self.best_score = self.worst_score
    self.best_exp = None
    self.best_params = None
    self.extra_child_args = []

    self.parameter_setup()
    self.restore(opt.restore_session)

  def checkpoint_ref_setup(self):
    super().checkpoint_ref_setup()
    self.checkpoint_ref['best'] = ['best_exp', 'best_score', 'best_params']
    self.checkpoint_ref['results'] = ['results', 'exp_count']

  def parameter_setup(self):
    opt = self.opt
    if opt.param:
      param = importlib.import_module('mtl.meta.param.' + opt.param)
      self.param = param.Metaparam
      opt.search = opt.search.split('-')

      test_sample = self.init_sample()
      self.curr_params = test_sample
      self.best_params = test_sample
      self.n_params = test_sample.get_params(opt.search).nelement()

    else:
      # Not optimizing any parameters
      self.param = None
      self.n_params = 0

  def init_sample(self):
    return self.param(self.opt)

  def copy_sample(self, p, k=None):
    p_new = self.init_sample()
    p_new.update_params(k, p.get_params(k))
    return p_new

  def submit_cmd(self,
                 cmd_queue,
                 exp_id,
                 param=None,
                 extra_args=None,
                 extra_child_args=[],
                 worker_mode=None):
    if worker_mode is None:
      worker_mode = self.opt.worker_mode
    sub_exp_id = '%s/%s' % (self.opt.exp_id, exp_id)
    extra_child_args = self.extra_child_args + extra_child_args

    if worker_mode == 'cmd':
      tmp_cmd = ['python', 'main.py', '-e', sub_exp_id] + self.base_cmd
      if extra_args is not None:
        tmp_cmd += extra_args

      # Initialize sub experiment directory
      tmp_dir = self.exp_root_dir + '/' + sub_exp_id
      mkdir_p(tmp_dir)

      if param is not None:
        if param.is_command:
          # Add extra arguments to new experiment
          param_file = '%s/params.txt' % self.exp_dir
          param_cmd = param.arg.get_cmd()
          with open(param_file, 'a') as f:
            f.write('%d %s\n' % (self.exp_count, ' '.join(param_cmd)))
          tmp_cmd += param_cmd

        else:
          # Point new experiment to metaparameters to load
          param_dir = '%s/params/%d/%d' % (self.opt.exp_id,
                                           self.exp_count // 100,
                                           self.exp_count % 100)
          param_path = '%s/%s' % (self.exp_root_dir, param_dir)
          mkdir_p(param_path)
          torch.save({'metaparams': param.state_dict()},
                     '%s/snapshot_meta' % param_path)
          tmp_cmd += ['--metaparam_load', param_dir]

    elif worker_mode == 'debug':
      tmp_cmd = [sub_exp_id, param.state_dict()]

    else:
      raise ValueError('Undefined worker mode: %s' % worker_mode)

    cmd_queue.put((self.exp_count, worker_mode, tmp_cmd, extra_child_args))
    self.exp_count += 1

  def collect_batch_results(self, n, result_queue):
    opt = self.opt
    scores = np.zeros(n)
    result_idx_ref = np.zeros(n, int)

    for i in range(n):
      result = result_queue.get()
      self.results += [result]
      sample_id = int(result[0].split('/')[-1])
      result_idx_ref[sample_id] = len(self.results) - 1
      if result[1] is not None:
        scores[sample_id] = result[1]['score']
      else:
        scores[sample_id] = self.worst_score

    if not opt.maximize:
      scores = -scores
    return result_idx_ref, scores

  def run(self, cmd_queue, result_queue):
    return

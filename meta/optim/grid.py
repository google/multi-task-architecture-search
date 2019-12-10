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

from mtl.meta.optim import random_search
import numpy as np
import torch


def setup_extra_args(parser):
  parser.add_argument('--n_steps', type=str, default='4')


class MetaOptimizer(random_search.MetaOptimizer):

  def setup_samples(self):
    opt = self.opt

    if len(opt.n_steps) == 1:
      n_steps = [int(opt.n_steps)] * self.n_params
    else:
      n_steps = list(map(int, opt.n_steps.split('-')))

    lspaces = [np.linspace(0, 1, n) for n in n_steps]
    all_params = np.meshgrid(*lspaces)
    all_params = [p.flatten() for p in all_params]
    all_params = np.stack(all_params, 1)

    all_samples = []
    for p in all_params:
      sample = self.init_sample()
      sample.update_params(opt.search, torch.Tensor(p))
      all_samples += [sample]

    return all_samples

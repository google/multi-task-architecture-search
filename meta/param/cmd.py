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

"""Parameter management of command line options (for hyperparam tuning)."""
import os

from mtl.meta.param import manager
from mtl.util import calc
import torch


curr_dir = os.path.dirname(__file__)
config_dir = os.path.join(curr_dir, '../../config/cmd')


def setup_extra_args(parser):
  parser.add_argument('--param_init', type=str, default='random')


class Metaparam(manager.Metaparam):

  def __init__(self, opt, config_fn=None):
    super().__init__()

    if config_fn is None:
      config_fn = opt.cmd_config
    config_file = '%s/%s.txt' % (config_dir, config_fn)
    args = []
    with open(config_file) as f:
      for line in f:
        # arg name, default, min, max, log/linear
        vals = line[:-1].split(' ')
        for i in range(1, 4):
          vals[i] = float(vals[i])
        vals[-1] = int(vals[-1])
        args += [vals]

    self.arg = ArgManager(args, opt.param_init)
    self.is_command = True


class ArgManager(manager.ParamManager):

  def __new__(cls, args, param_init='default'):
    return super().__new__(cls, shape=[len(args)])

  def __init__(self, args, param_init='default'):
    super().__init__()
    self.arg_ref = args
    self.valid_range = [0, 1]
    self.data = self.get_default()
    if param_init == 'random':
      self.set_to_random()

  def get_default(self):
    vals = torch.zeros(self.shape)
    for i, ref in enumerate(self.arg_ref):
      vals[i] = calc.map_val(*ref[1:-1], invert=True)
    return vals

  def get_cmd(self):
    tmp_cmd = []
    for i, ref in enumerate(self.arg_ref):
      tmp_cmd += [ref[0]]
      val = calc.map_val(self.data[i], *ref[2:-1])
      if ref[-1]:
        tmp_cmd += ['%d' % int(val)]
      else:
        tmp_cmd += ['%.3g' % val]
    return tmp_cmd

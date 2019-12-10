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

"""Management of command line options."""

import argparse
import importlib
import os
import pickle
import sys

from mtl.config.opts import meta as meta_opts
from mtl.config.opts import train as train_opts


def get_exp_root_dir():
  """Defines path of experiment root directory."""

  # Change this to save experiments somewhere else
  tmp_dir = os.path.dirname(__file__)
  tmp_dir = tmp_dir.split('/')[:-3] + ['exp']
  return '/'.join(tmp_dir)


def mkdir_p(dirname):
  """Make directory and any necessary parent directories."""
  try:
    os.makedirs(dirname)
  except FileExistsError:
    pass


def suppress_output():
  """Suppress all further output to the console."""
  f = open(os.devnull, 'w')
  sys.stdout = f
  sys.stderr = f


def get_disp_opts(is_meta):
  """Defines which flags are displayed when starting an experiment."""
  if is_meta:
    return [
        'exp_id',
        'metaoptimizer',
        'worker_mode',
        'num_samples',
        'param',
        'search',
    ]
  else:
    return [
        'exp_id', 'task', 'model', 'dataset', 'optimizer', 'batchsize',
        'learning_rate'
    ]


def setup_init_args(parser):
  """Initial arguments that need to be parsed first."""
  parser.add_argument('-e', '--exp_id', type=str, default='default')
  parser.add_argument('-x', '--is_meta', action='store_true')
  parser.add_argument('--config', type=str, default='exps.dec')
  parser.add_argument('--gpu_choice', type=str, default='0')
  parser.add_argument('--suppress_output', type=int, default=0)
  parser.add_argument('--task_choice', type=str, default='1-2-3-4-5-6-7-8-9')
  parser.add_argument('--fixed_seed', type=int, default=0)

  # Options to continue/branch off existing experiment
  g = parser.add_mutually_exclusive_group()
  g.add_argument('-c', '--continue_exp', type=int, default=0)
  g.add_argument('--branch', default='')


def restore_flags(parser, init_flags):
  """Updates options to match cached values.

  When restoring options from a previously run experiment, we want to load all
  values and update any terms that have been manually changed on the command
  line (for example, when continuing with a new learning rate).

  Args:
    parser: Command line parser used to collect init_flags.
    init_flags: Current flags that will tell us whether or not to load previous
      values from another experiment.
  """

  init_flags.restore_session = None
  last_round = 0

  # Check if we need to load up a previous set of options
  if init_flags.continue_exp or init_flags.branch:
    if init_flags.continue_exp:
      tmp_exp_dir = init_flags.exp_id
    else:
      tmp_exp_dir = init_flags.branch

    tmp_exp_dir = '%s/%s' % (get_exp_root_dir(), tmp_exp_dir)

    # Load previous options
    with open(tmp_exp_dir + '/opts.p', 'rb') as f:
      flags = pickle.load(f)

    # Parse newly set options
    setup_parser(parser, init_flags, flags)
    new_flags, _ = parser.parse_known_args()

    # Check which flags have been manually set and update them
    opts = {}
    for val in sys.argv:
      if val == '--':
        break
      elif val and val[0] == '-':
        if val in flags.short_ref:
          tmp_arg = flags.short_ref[val]
        else:
          tmp_arg = val[2:]
        if tmp_arg in new_flags:
          opts[tmp_arg] = new_flags.__dict__[tmp_arg]

    if '--' in sys.argv:
      opts['unparsed'] = sys.argv[sys.argv.index('--'):]

    for opt in opts:
      flags.__dict__[opt] = opts[opt]

    flags.restore_session = tmp_exp_dir

    try:
      with open(tmp_exp_dir + '/last_round', 'r') as f:
        last_round = int(f.readline())
    except:
      pass

  else:
    flags = init_flags

  if 'num_rounds' in flags:
    flags.last_round = last_round + flags.num_rounds

  return flags


def add_extra_args(parser, files):
  """Add additional arguments defined in other files."""

  for f in files:
    m = importlib.import_module(f)
    if 'setup_extra_args' in m.__dict__:
      m.setup_extra_args(parser)


def setup_parser(parser, init_flags, ref_flags=None):
  """Setup appropriate arguments and defaults for command line parser."""

  # Load config file
  if ref_flags is None:
    cfg = importlib.import_module('mtl.config.' + init_flags.config)
    is_meta = cfg.is_meta if 'is_meta' in cfg.__dict__ else init_flags.is_meta
  else:
    cfg = importlib.import_module('mtl.config.' + ref_flags.config)
    is_meta = ref_flags.is_meta

  parser.set_defaults(is_meta=is_meta)

  if is_meta:
    # Set up meta optimization options
    meta_opts.setup_args(parser)
    extra_arg_files = [['mtl.meta.optim', 'metaoptimizer'],
                       ['mtl.meta.param', 'param']]
  else:
    # Set up network training options
    train_opts.setup_args(parser)
    extra_arg_files = [['mtl.models', 'model'],
                       ['mtl.util.datasets', 'dataset'], ['mtl.train', 'task']]

  if ref_flags is None:
    for _, k in extra_arg_files:
      if k in cfg.option_defaults:
        parser.set_defaults(**{k: cfg.option_defaults[k]})
    ref_flags, _ = parser.parse_known_args()

  # Add additional arguments
  add_extra_args(parser, [
      '%s.%s' % (d, ref_flags.__dict__[k])
      for d, k in extra_arg_files
      if ref_flags.__dict__[k]
  ])

  # Update options/defaults according to config file
  if 'option_defaults' in cfg.__dict__:
    parser.set_defaults(**cfg.option_defaults)


def parse_command_line():
  """Parse command line and set up experiment options.

  Returns:
    An object with all options stored as attributes.
  """

  parser = argparse.ArgumentParser()
  setup_init_args(parser)
  init_flags, _ = parser.parse_known_args()

  # Check whether to restore previous options
  flags = restore_flags(parser, init_flags)

  if flags.restore_session is None:
    setup_parser(parser, init_flags)
    flags, unparsed = parser.parse_known_args()
    flags.unparsed = unparsed
    flags.restore_session = None
    if not flags.is_meta:
      flags.last_round = flags.num_rounds
    flags.short_ref = {
        a.option_strings[0]: a.option_strings[1][2:]
        for a in parser._actions
        if len(a.option_strings) == 2
    }

  # Save options
  flags.exp_root_dir = get_exp_root_dir()
  flags.data_dir = ''
  flags.exp_dir = '%s/%s' % (flags.exp_root_dir, flags.exp_id)
  mkdir_p(flags.exp_dir)
  with open('%s/opts.p' % flags.exp_dir, 'wb') as f:
    pickle.dump(flags, f)

  if not flags.is_meta:
    flags.iters = {'train': flags.train_iters}
    flags.drop_learning_rate = []
    if flags.drop_lr_iters:
      flags.drop_learning_rate = list(map(int, flags.drop_lr_iters.split('-')))

  if flags.suppress_output:
    suppress_output()
  print('---------------------------------------------')
  for tmp_opt in get_disp_opts(flags.is_meta):
    print('{:15s}: {}'.format(tmp_opt, flags.__dict__[tmp_opt]))
  print('---------------------------------------------')

  return flags


if __name__ == '__main__':
  print(parse_command_line())

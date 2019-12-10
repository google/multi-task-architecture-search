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

"""Class for managing training and meta-optimization sessions."""
import subprocess
import torch


class SessionManager():
  """Base class for managing experiments."""

  def __init__(self, opt, ds=None, dataloaders=None):
    self.opt = opt
    self.ds = ds
    self.dataloaders = dataloaders
    self.score = 0

    # Absolute paths for experiment information
    self.exp_root_dir = opt.exp_root_dir
    self.exp_dir = opt.exp_dir

    self.checkpoint_ref_setup()
    self.tensorboard_setup()

  def checkpoint_ref_setup(self):
    self.checkpoint_ref = {'extra': ['score']}

  def tensorboard_setup(self):
    return

  def setup_optimizer(self, params, multi=False):
    """Initialize parameter optimizer.

    Args:
      params: Parameters to optimize.
      multi: Optimizing multiple sets of parameters. If true, keep track of
        gradient statistics separately.
    """
    opt = self.opt
    optim_choice = opt.optimizer
    optim_fn = torch.optim.__dict__[optim_choice]

    # Setup optimizer arguments
    optim_kargs = {'lr': opt.learning_rate}
    if optim_choice == 'SGD':
      if 'momentum' in opt:
        optim_kargs['momentum'] = opt.momentum
        optim_kargs['nesterov'] = True
      if 'weight_decay' in opt:
        optim_kargs['weight_decay'] = opt.weight_decay

    elif optim_choice == 'RMSprop':
      optim_kargs['momentum'] = 0.
      optim_kargs['eps'] = 0.1

    # Initialize optimizers
    if multi:
      for i, p in enumerate(params):
        self.__dict__['optim_%d' % i] = optim_fn(p, **optim_kargs)
      self.checkpoint_ref['optim'] = [
          'optim_%d' % i for i in range(self.num_tasks)
      ]

    else:
      self.optim = optim_fn(params, **optim_kargs)
      self.checkpoint_ref['optim'] = ['optim']

  def checkpoint(self, path, key_ref=None, groups=None, action='save'):
    """Manage loading and restoring experiment checkpoints.

    Args:
      path: File path to save/load checkpoint.
      key_ref: Dictionary listing all parts of session to save.
      groups: Which subset of key_ref to save.
      action: 'save' or 'load'
    """
    if key_ref is None:
      key_ref = self.checkpoint_ref
    if groups is None:
      groups = key_ref.keys()

    for group in groups:
      if action == 'save':
        to_, from_ = {}, self.__dict__
      elif action == 'load':
        to_, from_ = self.__dict__, torch.load('%s_%s' % (path, group))

      for k in key_ref[group]:
        try:
          # Check whether or not to use a state_dict
          tmp_obj = from_[k] if action == 'save' else to_[k]
          use_state_dict = callable(getattr(tmp_obj, 'state_dict'))
        except AttributeError:
          use_state_dict = False

        if use_state_dict:
          if action == 'save':
            to_[k] = from_[k].state_dict()
          elif action == 'load':
            kargs = {'strict': False} if 'model' in k else {}
            to_[k].load_state_dict(from_[k], **kargs)

        else:
          to_[k] = from_[k]

      if action == 'save':
        torch.save(to_, '%s_%s' % (path, group))

  def save(self, path, key_ref=None, groups=None):
    self.checkpoint(path, key_ref, groups, 'save')

  def load(self, path, key_ref=None, groups=None):
    self.checkpoint(path, key_ref, groups, 'load')

  def restore(self, path, groups=None):
    if path:
      print('Restoring previous session... (%s/snapshot)' % path)
      self.load(path + '/snapshot', groups=groups)

  def clear_checkpoints(self, path, groups=['model', 'optim']):
    for g in groups:
      try:
        subprocess.call(['rm', path + '/snapshot_%s' % g])
      except Exception as e:
        print('Error clearing snapshot %s:' % g, repr(e))

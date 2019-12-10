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

from mtl.meta.param import partition
from mtl.util import calc
from mtl.util import session_manager
from mtl.util import tensorboard_manager
import numpy as np
import torch


class Task(session_manager.SessionManager):
  """Basic network training manager."""

  def __init__(self, opt, ds, dataloaders):
    super().__init__(opt, ds, dataloaders)
    self.opt = opt
    self.ds = ds
    self.dataloaders = dataloaders
    self.is_training = True
    splits = ['train', 'valid']

    # Task reference information
    self.num_tasks = len(ds['train'])
    self.task_idx_ref = list(map(int, opt.task_choice.split('-')))

    self.targets = ds['train'][0].targets
    task_ds_size = [len(d) for d in ds['train']]
    self.task_ds_size = task_ds_size
    self.task_proportion = [t / sum(task_ds_size) for t in task_ds_size]

    self.task_num_out = [d.num_out for d in ds['train']]
    self.task_trained_steps = [0 for _ in range(self.num_tasks)]
    self.global_trained_steps = 0

    # Initialize running averages for logging
    self.log = {}
    for k in self.to_track:
      self.log[k] = [
          {s: 0 for s in splits} for _ in range(self.num_tasks)
      ]
      self.log['%s_history' % k] = [
          {s: [] for s in splits} for _ in range(self.num_tasks)
      ]

    self.score = 0

    # Set up dataset iterators
    self.iters = {s: [] for s in splits}
    for split in self.iters:
      for task_idx in range(self.num_tasks):
        self.iters[split] += [iter(self.dataloaders[split][task_idx])]

    # Set up metaparameters
    self.setup_metaparams(opt, ds)
    if opt.metaparam_load:
      del self.to_load[self.to_load.index('meta')]
      metaparam_path = '%s/%s/snapshot' % (opt.exp_root_dir, opt.metaparam_load)
      print('Loading parameters from... (%s)' % metaparam_path)
      self.load(metaparam_path, groups=['meta'])

    # Set up model and optimization and loss
    self.setup_model(opt, ds)
    self.setup_optimizer(self.model.net_parameters, True)
    self.loss_fn = torch.nn.CrossEntropyLoss()

    # Check for a fixed curriculum
    if opt.curriculum == 'fixed':
      print('Loading fixed curriculum.')
      self.fixed_curriculum = ds['train'][0].load_fixed_curriculum()

    # Load pretrained model weights, restore previous checkpoint
    if opt.pretrained:
      tmp_path = '%s/%s/snapshot_model' % (opt.exp_root_dir, opt.pretrained)
      print('Loading pretrained model weights from:', tmp_path)
      pretrained = torch.load(tmp_path)
      self.model.load_state_dict(pretrained['model'], strict=False)

    self.restore(opt.restore_session, self.to_load)

  def tensorboard_setup(self):
    self.to_track = ['network_loss', 'accuracy', 'decathlon']
    self.task_names = [d.task_name for d in self.ds['train']]
    self.tb = tensorboard_manager.TBManager(self.exp_dir, self.task_names,
                                            self.to_track)

  def checkpoint_ref_setup(self):
    # Define key/value pairs for checkpoint management
    self.checkpoint_ref = {
        'model': ['model'],
        'meta': ['metaparams'],
        'extra': ['log', 'score', 'task_trained_steps', 'global_trained_steps']
    }
    self.to_load = list(self.checkpoint_ref.keys())

  def setup_metaparams(self, opt, ds):
    if opt.metaparam == 'partition':
      self.metaparams = partition.Metaparam(opt)
      self.masks = None
      self.checkpoint_ref['model'] += ['masks']

    else:
      self.metaparams = None

  def setup_model(self, opt, ds):
    model = importlib.import_module('mtl.models.' + opt.model)

    if opt.metaparam == 'partition':
      if opt.restore_session is not None:
        # Restore session snapshot
        print('Restoring previous masks... (%s/snapshot)' % opt.restore_session)
        checkpoint = torch.load('%s/snapshot_model' % opt.restore_session)
        self.masks = checkpoint['masks']

      self.model = model.initialize(
          opt, ds, metaparams=self.metaparams, masks=self.masks)
      self.masks = [self.model.resnet.mask_ref, self.model.grad_mask_ref]

    else:
      self.model = model.initialize(opt, ds)

  def cuda(self):
    self.model.cuda()

  def set_train_mode(self, train_flag):
    self.is_training = train_flag
    if train_flag:
      self.model.train()
    else:
      self.model.eval()

  def get_log_vals(self, split, task_idx):
    return {k: self.log[k][task_idx][split] for k in self.to_track}

  def update_log(self, k, v):
    tmp_log = self.log[k][self.task_idx]
    if self.step == 0:
      tmp_v = v
    else:
      tmp_v = calc.running_avg(tmp_log[self.split], v)
    tmp_log[self.split] = tmp_v

  def sample_task(self):
    task_idxs = np.arange(self.num_tasks)
    curr = self.opt.curriculum

    if self.split == 'train':
      if curr == 'fixed':
        # Fixed training curriculum
        return self.fixed_curriculum[self.step % len(self.fixed_curriculum)]

      elif curr == 'uniform':
        # Uniformly iterate through tasks
        return self.step % self.num_tasks

      elif curr == 'proportional':
        # Sample tasks proportional to dataset size
        return np.random.choice(task_idxs, p=self.task_proportion)

      elif curr == 'train_accuracy':
        # Sample tasks based on relative training accuracies
        train_acc = np.array([d['train'] for d in self.log['accuracy']])
        train_acc = np.log(1 - train_acc + self.opt.curriculum_bias)
        task_dist = calc.softmax(train_acc / self.opt.temperature)
        return np.random.choice(task_idxs, p=task_dist)

      else:
        # Undefined, raise error
        raise ValueError('Undefined task curriculum: %s' % curr)

    else:
      # Validation (order doesn't matter, just hit all validation samples)
      return self.opt.valid_iter_ref[self.step % self.opt.iters['valid']]

  def get_next_sample(self, split, task_idx):
    reset_iter = False
    try:
      sample = self.iters[split][task_idx].next()
    except StopIteration:
      reset_iter = True

    if reset_iter:
      self.iters[split][task_idx] = iter(self.dataloaders[split][task_idx])
      sample = self.iters[split][task_idx].next()

    inp = sample['img'].cuda()
    label = sample['label'].view(-1).cuda()
    idxs = sample['index']

    return inp, label, idxs

  def run(self, split, step):
    self.step = step
    self.split = split

    # Sample task
    task_idx = self.sample_task()
    self.task_idx = task_idx
    self.curr_task = self.ds['train'][task_idx].task_name

    # Get samples + model output
    inp, label, idxs = self.get_next_sample(split, task_idx)
    pred = self.model(inp, task_idx, split, step)

    # Calculate loss
    _, class_preds = torch.max(pred, 1)
    accuracy = class_preds.eq(label).float().mean()
    dec_score, _ = calc.decathlon_score(
        accuracy.data, task_idxs=[self.task_idx_ref[task_idx]])
    self.net_loss = self.loss_fn(pred, label)

    # Track accuracy and cache predictions
    if split == 'valid':
      self.valid_accuracy_track[task_idx] += [accuracy.data.item()]
      for batch_idx, tmp_idx in enumerate(idxs):
        self.prediction_ref[task_idx][
            tmp_idx.item()] = class_preds.data[batch_idx].item()

    self.update_log('accuracy', accuracy.data.item())
    self.update_log('network_loss', self.net_loss.data.item())
    self.update_log('decathlon', dec_score)
    self.score = np.array([d['valid'] for d in self.log['accuracy']]).mean()

    self.global_trained_steps += 1
    self.task_trained_steps[task_idx] += 1

  def update_weights(self):
    opt = self.opt
    t_idx = self.task_idx

    # Set up optimizer and learning rate
    net_optimizer = self.__dict__['optim_%d' % t_idx]
    lr = opt.learning_rate
    for p in net_optimizer.param_groups:
      p['lr'] = lr

    # Calculate gradients
    net_optimizer.zero_grad()
    self.net_loss.backward()

    if opt.clip_grad:
      p = self.model.net_parameters[t_idx]
      torch.nn.utils.clip_grad_norm(p, opt.clip_grad)

    if opt.metaparam == 'partition' and opt.mask_gradient:
      # Loop through parameters and multiply by appropriate mask
      for m, bw_masks in self.model.bw_ref:
        bw_mask = bw_masks[t_idx].cuda()
        m.weight.grad *= bw_mask
        if 'bias' in m._parameters and m.bias is not None:
          m.bias.grad *= bw_mask[:, 0]

    # Do weight update
    net_optimizer.step()

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

"""Feature partitioning parameter management."""
from mtl.meta.param import manager
from mtl.util import calc
import numpy as np
import torch


def setup_extra_args(parser):
  parser.add_argument('--param_init', type=str, default='random')
  parser.add_argument('--rand_type', type=str, default='restrict')
  parser.add_argument('--restrict_amt', type=float, default=.5)
  parser.add_argument('--num_unique', type=int, default=1)
  parser.add_argument('--mask_gradient', type=int, default=1)
  parser.add_argument('--share_amt', type=float, default=.75)


def prepare_masks(p, num_feats):
  # Prepare masks
  num_tasks = p.num_tasks
  num_unique = p.num_unique
  mask_ref = [[{} for _ in range(num_unique)] for _ in range(num_tasks)]
  grad_mask_ref = [[{} for _ in range(num_unique)] for _ in range(num_tasks)]

  for t in range(num_tasks):
    for u in range(num_unique):
      for f in num_feats:
        # Forward mask
        tmp_mask = p.get_task_mask(t, f, m_idx=0, p_idx=u)
        mask_ref[t][u][f] = torch.Tensor(tmp_mask).view(1, -1, 1, 1)

        # Backward mask
        tmp_grad_mask = p.get_task_mask(t, f, m_idx=1, p_idx=u)
        grad_mask_ref[t][u][f] = torch.Tensor(tmp_grad_mask).view(-1, 1, 1, 1)

  return mask_ref, grad_mask_ref


def masked_cnv(ref, cnv, unq_idx=0):
  num_f = cnv.out_channels
  return lambda x, y: cnv(x) * ref.mask_ref[y][unq_idx][num_f].cuda()


class Metaparam(manager.Metaparam):

  def __init__(self, opt):
    super().__init__()
    self.partition = PartitionManager(opt)


class PartitionManager(manager.ParamManager):

  def __new__(cls, opt):
    return super().__new__(cls, shape=None)

  def __init__(self, opt):
    super().__init__()
    self.param_init = opt.param_init
    self.rand_type = opt.rand_type
    self.restrict_amt = opt.restrict_amt
    self.share_amt = opt.share_amt
    self.valid_range = [0, 1]
    self.num_tasks = len(opt.task_choice.split('-'))
    self.num_unique = opt.num_unique
    shape = [self.num_unique, 2, self.num_tasks, self.num_tasks]
    self.data = torch.Tensor(*shape)
    self.data = self.get_default()
    self.mask = {}
    self.set_partition(opt.param_init)
    self.ref_triu_idxs = np.triu(np.ones(shape[1:])) != 0

  def set_partition(self, share_type=None):
    if share_type is None:
      share_type = self.param_init
    tmp_mat = None
    n = self.num_tasks
    ones = torch.ones((2, n, n))
    eye = torch.eye(n)

    if share_type == 'share_all':
      tmp_mat = ones

    elif share_type == 'share_fixed':
      share_pct = self.share_amt
      indiv_pct = share_pct + (1 - share_pct) / n
      share_val = (n - 2) / (n - 1)
      tmp_mat = ones * share_val
      tmp_mat[0] += eye * (indiv_pct - share_val)
      tmp_mat[1] = 1

    elif share_type == 'share_fwd_only':
      tmp_mat = ones
      tmp_mat[1] = eye * (1. / n)

    elif share_type == 'independent':
      tmp_mat = ones
      tmp_mat[0] = eye * (1. / n)
      tmp_mat[1] = eye

    elif share_type == 'random':
      self.set_to_random()

    else:
      raise ValueError('Undefined partition setting: %s' % share_type)

    if tmp_mat is not None:
      for i in range(self.num_unique):
        self.data[i] = tmp_mat

  def ignore_backward_mask(self):
    for d in self.data:
      d[1].fill_(1)

  def get_default(self):
    return torch.ones(self.shape)

  def set_to_random(self):
    super().set_to_random()
    self.reset_masks()

  def random_sample(self, rand_type=None):
    if rand_type is None:
      rand_type = self.rand_type
    tmp_mat = torch.rand(self.shape)
    n = self.num_tasks
    eye = torch.eye(n)

    if rand_type == 'restrict':
      mutate = torch.randn(n, n) * .2
      for m in tmp_mat:
        m[0] = eye * (self.restrict_amt + mutate) + (1 - eye) * m[0]
        m[1] = eye * (1 + torch.randn(n, n) * .15) + (1 - eye) * m[1]
      tmp_mat = tmp_mat.clamp(.05, 1)

    return tmp_mat

  def preprocess_mat(self, mat, diag=None):
    # Preserve diagonal and make matrix symmetric
    if diag is None:
      diag = mat.diag()
    mat = torch.triu(mat, 1)
    mat = diag.diag() + mat + mat.t()

    return mat.clamp(0, 1)

  def find_masks(self, p=None, n_feats=100, n_iters=100, p_idx=0):
    # Convert raw parameterization
    if p is None:
      p = self.data[p_idx]

    if self.param_init == 'share_fixed':
      amt_per_task = int(
          np.floor(n_feats * (1 - self.share_amt) / self.num_tasks))
      amt_shared = n_feats - (amt_per_task * self.num_tasks)
      masks = []
      for i in range(self.num_tasks):
        tmp_mask = np.zeros(n_feats)
        tmp_mask[i * amt_per_task:(i + 1) * amt_per_task] = 1
        tmp_mask[-amt_shared:] = 1
        masks += [tmp_mask]
      mask_f = np.stack(masks)
      mask_b = mask_f
      return np.stack([mask_f, mask_b], 0)

    else:
      return calc.find_masks(p, n_feats, n_iters)

  def init_mask(self, n_feats=100, n_iters=100):
    if n_feats not in self.mask:
      self.mask[n_feats] = [
          self.find_masks(n_feats=n_feats, n_iters=n_iters, p_idx=p_idx)
          for p_idx in range(self.num_unique)
      ]
      self.mask[n_feats] = [m.copy() for m in self.mask[n_feats]]

  def get_task_mask(self, task_idx, n_feats, p_idx=0, m_idx=0):
    self.init_mask(n_feats)
    return self.mask[n_feats][p_idx][m_idx][task_idx]

  def reset_masks(self):
    self.mask = {}

  def mutate(self, data=None, delta=.1):
    v_min, v_max = self.valid_range
    update_self = data is None
    if update_self:
      data = self.data

    if isinstance(data, torch.Tensor):
      data = data.clone()
      data += torch.randn(data.shape) * delta
      data = data.clamp(v_min, v_max)
    else:
      data = data.copy()
      data += np.random.randn(*data.shape) * delta
      data = data.clip(v_min, v_max)

    if update_self:
      self.data = data
    return data

  def get_task_parameter_mask(self, task_idx):
    # Return a binary mask indicating which parameters are directly
    # associated with a particular task.
    v = np.zeros_like(self.ref_triu_idxs)
    v[:, :, task_idx] = True
    v[:, task_idx, :] = True
    v = (v * self.ref_triu_idxs).astype(int)
    tmp_p = np.expand_dims(v, 0).repeat(self.num_unique, 0)

    return tmp_p.flatten()

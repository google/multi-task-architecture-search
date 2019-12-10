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
import torch
import torch.nn.functional as F


def running_avg(x, y, k=.99):
  return k * x + (1 - k) * y


def softmax(x, d=-1):
  tmp = np.exp(np.array(x))
  return tmp / tmp.sum(axis=d, keepdims=True)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
  return np.log(x / (1 - x))


def dist(a, b):
  return np.linalg.norm(a - b)


def smooth_arr(arr, window=3):
  to_flatten = False
  if arr.ndim == 1:
    to_flatten = True
    arr = np.expand_dims(arr, 1)

  pad = window // 2
  tmp_arr = F.pad(
      torch.unsqueeze(torch.Tensor(arr.T), 0), [pad, pad], mode='reflect')
  tmp_arr = np.array(F.avg_pool1d(tmp_arr, window, stride=1).data)
  tmp_arr = tmp_arr[0].T

  if to_flatten:
    tmp_arr = tmp_arr[:, 0]

  return tmp_arr


def decathlon_score(scores, task_idxs=None):
  if task_idxs is None:
    task_idxs = [i for i in range(10)]
  baseline_err = 1 - np.array([
      59.87, 60.34, 82.12, 92.82, 55.53, 97.53, 81.41, 87.69, 96.55, 51.20
  ]) / 100
  baseline_err = baseline_err[task_idxs]
  num_tasks = len(task_idxs)

  max_err = 2 * baseline_err
  gamma_vals = np.ones(num_tasks) * 2
  alpha_vals = 1000 * (max_err)**(-gamma_vals)

  err = 1 - scores
  if num_tasks == 1:
    err = [err]

  all_scores = []
  for i in range(num_tasks):
    all_scores += [alpha_vals[i] * max(0, max_err[i] - err[i])**gamma_vals[i]]
  return sum(all_scores), all_scores


def rescale(x, min_val, max_val, invert=False):
  if not invert:
    return x * (max_val - min_val) + min_val
  else:
    return (x - min_val) / (max_val - min_val)


def pow10(x, min_val, max_val, invert=False):
  log_fn = np.log if type(x) is float else torch.log

  if not invert:
    return 10**rescale(x,
                       np.log(min_val) / np.log(10),
                       np.log(max_val) / np.log(10))
  else:
    return rescale(
        log_fn(x) / np.log(10),
        np.log(min_val) / np.log(10),
        np.log(max_val) / np.log(10), invert)


def map_val(x, min_val, max_val, scale='linear', invert=False):
  if scale == 'log':
    map_fn = pow10
  elif scale == 'linear':
    map_fn = rescale
  return map_fn(x, min_val, max_val, invert)


def reverse_tensor(t, dim):
  return t.index_select(dim, torch.arange(t.shape[dim] - 1, -1, -1).long())


def convert_mat_aux(m, d, min_, max_, invert=False):
  if invert:
    m = (m - min_) / (max_ - min_ + 1e-5)
  else:
    m = m * (max_ - min_) + min_
  m = np.triu(m, 1)
  return m + m.T + np.diag(d)


def convert_mat(mat, invert=False):
  if mat.dim() == 3:
    # mat is 2 x n x n
    # where mat[0] is the forward matrix, and mat[1] is the backward one
    mat = np.array(mat)

    # Convert forward matrix
    d_f = mat[0].diagonal()
    min_ = np.maximum(0, np.add.outer(d_f, d_f) - 1)
    max_ = np.minimum.outer(d_f, d_f)
    m_f = convert_mat_aux(mat[0], d_f, min_, max_, invert=invert)

    # Convert backward matrix
    d_b = mat[1].diagonal()
    if not invert:
      d_b = d_b * d_f
    tmp_m = mat[0] if invert else m_f
    min_ = np.maximum(0,
                      np.add.outer(d_b, d_b) - np.add.outer(d_f, d_f) + tmp_m)
    max_ = np.minimum(tmp_m, np.minimum.outer(d_b, d_b))
    if invert:
      d_b = d_b / d_f
    m_b = convert_mat_aux(mat[1], d_b, min_, max_, invert=invert)

    tmp_mat = np.stack([m_f, m_b], 0)
    tmp_mat = np.round(tmp_mat * 1000) / 1000
    return torch.Tensor(tmp_mat)

  else:
    result = [convert_mat(m, invert=invert) for m in mat]
    return torch.stack(result)


def mask_solver(p, n_iters=10, n_feats=100, filt=None):
  pw = (p * n_feats + 1e-3).astype(int)
  diag = pw.diagonal()
  n_tasks = p.shape[0]
  all_idxs = np.arange(n_feats)

  mask = np.zeros((n_tasks, n_feats))
  mask[0][:pw[0, 0]] = 1

  p1_ = pw / np.maximum(1, np.outer(diag, diag))
  p2_ = -pw / np.maximum(1, np.outer(n_feats - diag, diag))
  p2_ += 1 / np.maximum(1, (n_feats - diag.reshape(-1, 1)))

  for curr_t in range(1, n_tasks):
    prob_dist = np.ones(n_feats)
    if filt is not None:
      prob_dist *= filt[curr_t]

    for cmp_t in range(curr_t):
      prob_dist *= p1_[cmp_t, curr_t] * mask[cmp_t] + p2_[cmp_t, curr_t] * (
          1 - mask[cmp_t])

    if prob_dist.sum() == 0:
      prob_dist = np.ones(n_feats)

    prob_dist /= prob_dist.sum()

    scores = np.zeros((n_iters, curr_t))
    best_score_dist = 999
    n_to_choose = min((prob_dist > 0).sum(), diag[curr_t])
    if n_to_choose > 0:
      for i in range(n_iters):
        sample_idxs = np.random.choice(
            all_idxs, n_to_choose, replace=False, p=prob_dist)
        tmp_row = np.zeros(n_feats)
        tmp_row[sample_idxs] = 1

        scores[i] = np.dot(mask[:curr_t], tmp_row)
        score_dist = np.linalg.norm(scores[i] - pw[:curr_t, curr_t])
        if score_dist < best_score_dist:
          best_score_dist = score_dist
          mask[curr_t] = tmp_row

  return mask


def find_masks(p, n_feats=100, n_iters=8, return_scores=True):
  # Convert raw parameterization
  p = np.array(convert_mat(p))
  kargs = {'n_feats': n_feats, 'n_iters': n_iters}
  mask_f = mask_solver(p[0], **kargs)
  mask_b = mask_solver(p[1], filt=mask_f, **kargs)

  # Sort mask channels
  count_row_f = mask_f.sum(0, keepdims=True)
  count_row_b = mask_b.sum(0, keepdims=True)
  tmp_mask = np.concatenate([mask_b, count_row_b, mask_f, count_row_f], 0)
  tmp_idx = np.lexsort(tmp_mask)
  tmp_mask = tmp_mask[:, tmp_idx]

  n = mask_b.shape[0]
  mask_b = tmp_mask[:n]
  mask_f = tmp_mask[n + 1:2 * n + 1]
  masks = np.stack([mask_f, mask_b], 0)

  return masks

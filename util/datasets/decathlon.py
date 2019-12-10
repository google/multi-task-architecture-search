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

"""Visual Decathlon dataset reference file."""
import json
import os
import pickle

import imageio
from mtl.third_party.cutout import cutout
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


def setup_extra_args(parser):
  parser.add_argument('--input_res', type=int, default=72)
  parser.add_argument('--subsample_validation', type=int, default=0)


task_names = [
    'imagenet12', 'aircraft', 'cifar100', 'daimlerpedcls', 'dtd', 'gtsrb',
    'vgg-flowers', 'omniglot', 'svhn', 'ucf101'
]


def collect_and_parse_annot(annot_dir, task_name, split):
  # Collect image paths, img_ids, cat_id, task_specific_cat, ht, wd
  ann = {}
  key_map = {'train': 'train', 'valid': 'val', 'test': 'test_stripped'}
  split = key_map[split]

  annot_fn = '%s/annotations/%s_%s.json' % (annot_dir, task_name, split)
  with open(annot_fn, 'r') as f:
    annot = json.load(f)

  num_imgs = len(annot['images'])
  keys = ['path', 'ht', 'wd', 'label', 'cat_id', 'img_id']
  ann = {k: np.zeros(num_imgs, int) for k in keys[1:]}
  ann[keys[0]] = []

  num_cats = len(annot['categories'])
  ann['num_cats'] = num_cats
  ann['cat_id_ref'] = np.zeros(num_cats, int)
  ann['cat_label_ref'] = []
  for i in range(num_cats):
    ann['cat_id_ref'][i] = annot['categories'][i]['id']
    ann['cat_label_ref'] += [annot['categories'][i]['name']]

  ann['num_imgs'] = num_imgs
  for i in range(num_imgs):
    im_annot = annot['images'][i]
    ann['path'] += [im_annot['file_name']]
    ann['ht'][i] = im_annot['height']
    ann['wd'][i] = im_annot['width']
    if 'test' in split:
      # For test images, put in all zeros as filler
      ann['img_id'][i] = im_annot['id']
      ann['cat_id'][i] = 0
      ann['label'][i] = 0
    else:
      gt_annot = annot['annotations'][i]
      ann['img_id'][i] = gt_annot['image_id']
      ann['cat_id'][i] = gt_annot['category_id']
      ann['label'][i] = gt_annot['category_id'] % 1e5 - 1

  return ann


def combine_annot(dt, dv):
  new_annot = {}
  for k in ['ht', 'wd', 'label', 'cat_id', 'img_id']:
    new_annot[k] = np.concatenate([dt[k], dv[k]], 0)
  new_annot['path'] = dt['path'] + dv['path']
  new_annot['num_imgs'] = dt['num_imgs'] + dv['num_imgs']
  for k in ['num_cats', 'cat_id_ref', 'cat_label_ref']:
    new_annot[k] = dt[k]
  return new_annot


def get_annot_from_idxs(annot, idxs):
  new_annot = {}
  for k in ['ht', 'wd', 'label', 'cat_id', 'img_id']:
    new_annot[k] = annot[k][idxs]
  new_annot['path'] = [annot['path'][i] for i in idxs]
  new_annot['num_imgs'] = len(idxs)
  for k in ['num_cats', 'cat_id_ref', 'cat_label_ref']:
    new_annot[k] = annot[k]
  return new_annot


class DecathlonDataset(Dataset):

  def __init__(self, opt, task_idx, is_train, annot=None, augment=None):
    if augment is None:
      augment = is_train
    task_name = task_names[task_idx]

    self.data_dir = opt.data_dir
    self.task_idx = task_idx
    self.task_name = task_name
    self.is_train = is_train
    self.targets = np.array(
        [59.87, 60.34, 82.12, 92.82, 55.53, 97.53, 81.41, 87.69, 96.55, 51.20])

    if annot is None:
      if is_train or opt.validate_on_train:
        # Load training annotations
        annot = collect_and_parse_annot(opt.data_dir, task_name, 'train')
        if opt.train_on_valid:
          valid_annot = collect_and_parse_annot(opt.data_dir, task_name,
                                                'valid')
          annot = combine_annot(annot, valid_annot)

      else:
        # Load validation/test annotations
        annot = collect_and_parse_annot(opt.data_dir, task_name,
                                        'test' if opt.use_test else 'valid')

        if opt.subsample_validation and not opt.use_test:
          num_valid = annot['num_imgs']
          if task_name != 'imagenet12':
            # Subsample to 3k or less (indices chosen randomly)
            tmp_filename = '%s/%s_valid_subset.npy' % (opt.data_dir, task_name)
            try:
              tmp_idxs = np.load(tmp_filename)
            except:
              tmp_idxs = np.random.permutation(np.arange(annot['num_imgs']))
              tmp_idxs = tmp_idxs[:3000]
              np.save(tmp_filename, tmp_idxs)
          else:
            # Subsample ImageNet validation uniformly by factor of 10
            tmp_idxs = np.arange(0, num_valid, 10)

          annot = get_annot_from_idxs(annot, tmp_idxs)

    self.num_out = annot['num_cats']
    self.num_imgs = annot['num_imgs']
    self.annot = annot

    # Setup image transformation
    no_flip = [5, 7, 8]
    scale_factor = 1.0

    if augment:
      tmp_transform = [
          transforms.Resize(int(opt.input_res * scale_factor)),
          transforms.RandomCrop(opt.input_res, padding=4),
          transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
      ]
      if task_idx not in no_flip:
        # Ignore datasets like SVHN (where left/right matters for digits)
        tmp_transform += [transforms.RandomHorizontalFlip()]

      tmp_transform = transforms.Compose(tmp_transform)
    else:
      tmp_transform = transforms.Compose([
          transforms.Resize(int(opt.input_res * scale_factor)),
          transforms.CenterCrop(opt.input_res)
      ])

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    with open(opt.data_dir + '/decathlon_mean_ref.p', 'rb') as f:
      # Precomputed mean/variance of decathlon images
      ref_mean_std = pickle.load(f)
    ds_mean = ref_mean_std['mean'][self.task_name]
    ds_std = ref_mean_std['std'][self.task_name]
    normalize = transforms.Normalize(mean=ds_mean, std=ds_std)

    self.transform = transforms.Compose(
        [to_pil, tmp_transform, to_tensor, normalize])
    if augment:
      self.transform.transforms.append(
          cutout.Cutout(n_holes=1, length=opt.input_res // 4))

  def load_fixed_curriculum(self):
    return np.load(self.data_dir + '/dec_order.npy')

  def load_image(self, idx):
    imgpath = '%s/%s' % (self.data_dir, self.annot['path'][idx])
    tmp_im = imageio.imread(imgpath).astype(float) / 255
    if tmp_im.ndim == 2:
      tmp_im = np.tile(np.expand_dims(tmp_im, 2), [1, 1, 3])

    return tmp_im

  def __len__(self):
    return self.num_imgs

  def __getitem__(self, idx):
    img = self.load_image(idx)
    img = torch.Tensor(img).permute(2, 0, 1)  # HWC --> CHW
    return {
        'img': self.transform(img),
        'label': self.annot['label'][idx],
        'index': idx
    }


def initialize(opt):
  if not opt.data_dir:
    # Default data directory
    curr_dir = os.path.dirname(__file__)
    opt.data_dir = os.path.join(curr_dir, '../../../data/decathlon')

  datasets = {'train': [], 'valid': []}
  dataloaders = {'train': [], 'valid': []}
  task_idxs = list(map(int, opt.task_choice.split('-')))

  # Check whether to run a different batchsize during validation
  valid_bs = opt.valid_batchsize if opt.valid_batchsize else opt.batchsize

  print('Training on:')
  valid_iter_ref = []
  for i, task_idx in enumerate(task_idxs):
    for split in datasets:
      is_train = split == 'train'
      datasets[split] += [DecathlonDataset(opt, task_idx, is_train)]
      dataloaders[split] += [
          DataLoader(
              datasets[split][i],
              batch_size=opt.batchsize if is_train else valid_bs,
              shuffle=False if opt.use_test else True,
              num_workers=opt.num_data_threads)
      ]
    print(task_names[task_idx],
          [len(datasets[split][-1]) for split in datasets])
    valid_iter_ref += [len(datasets['valid'][-1])]

  n_valid_samples = sum(valid_iter_ref)
  valid_iter_ref = [int(np.ceil(v / valid_bs)) for v in valid_iter_ref]
  opt.iters['valid'] = sum(valid_iter_ref)
  tmp_iter_ref = np.concatenate(
      [np.array([i] * v) for i, v in enumerate(valid_iter_ref)])
  opt.valid_iter_ref = np.random.permutation(tmp_iter_ref)
  opt.valid_iters = opt.iters['valid']
  print('%d validation samples available w/ batchsize %d (%d valid iters)' %
        (n_valid_samples, valid_bs, opt.iters['valid']))

  return datasets, dataloaders

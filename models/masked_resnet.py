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

from mtl.meta.param import partition
from mtl.models import resnet
import numpy as np


def setup_extra_args(parser):
  resnet.setup_extra_args(parser)
  partition.setup_extra_args(parser)
  parser.set_defaults(metaparam='partition', last_third_only=1)


class Net(resnet.Net):
  """ResNet that supports masking for feature partitioning."""

  def __init__(self, opt, ds, metaparams=None, masks=None):
    super(Net, self).__init__(opt, ds)

    self.meta = [metaparams]
    if self.meta[0] is None:
      self.meta = [partition.Metaparam(opt)]
    share = self.meta[0].partition
    self.num_unique = share.num_unique
    self.bw_ref = []

    layer_ref = self.resnet.get_layer_ref()
    num_feats = self.resnet.num_feat_ref
    if opt.bottleneck_ratio != 1:
      num_feats += [int(f * opt.bottleneck_ratio) for f in num_feats]

    if masks is None:
      # Prepare all masks
      self.resnet.mask_ref, self.grad_mask_ref = partition.prepare_masks(
          share, num_feats)
    else:
      self.resnet.mask_ref, self.grad_mask_ref = masks

    repeat_rate = int(np.ceil(len(layer_ref) / self.num_unique))
    unq_idx_ref = [i // repeat_rate for i in range(len(layer_ref))]

    # Convert all layers
    for l_idx, l in enumerate(layer_ref):
      unq_idx = unq_idx_ref[l_idx]
      tmp_m = self.resnet.get_module(l)
      l_name = l[-1]
      cnv = tmp_m._modules[l_name]

      if 'conv1' in l and (not opt.last_third_only or 'layer3' in l):
        # Apply mask to first convolution in ResBlock
        tmp_m.conv = partition.masked_cnv(self.resnet, cnv, unq_idx)

        # Save a reference for doing gradient masking
        bw_masks = [
            self.grad_mask_ref[i][unq_idx][cnv.out_channels]
            for i in range(self.num_tasks)
        ]
        self.bw_ref += [[cnv, bw_masks]]


def initialize(*args, **kargs):
  return Net(*args, **kargs)

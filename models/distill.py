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

import copy

from mtl.models import masked_resnet
from mtl.models import resnet
import torch


def setup_extra_args(parser):
  masked_resnet.setup_extra_args(parser)
  parser.add_argument('--last_layer_idx', type=int, default=1)
  parser.add_argument('--num_distill', type=int, default=2)
  parser.add_argument('--finetune_fc', type=int, default=0)
  parser.set_defaults(last_third_only=1)


class RefNet(resnet.Net):

  def __init__(self, opt, ds, sub_task_choice=None):
    super().__init__(opt, ds, sub_task_choice=None)
    self.last_layer_idx = opt.last_layer_idx

  def forward(self, x):
    x = self.resnet.pre_layers_conv(x)
    x = self.resnet.layer1(x)
    x = self.resnet.layer2(x)
    for i in range(self.last_layer_idx):
      x = self.resnet.layer3[i](x)

    return x


class Net(masked_resnet.Net):

  def __init__(self, opt, ds, metaparams=None, masks=None):
    super().__init__(opt, ds, metaparams, masks)

    # Layer distillation options
    self.last_layer_idx = opt.last_layer_idx
    self.num_distill = opt.num_distill
    self.finetune_fc = opt.finetune_fc

    # Reference performance for each task
    task_idxs = list(map(int, opt.task_choice.split('-')))
    self.task_low = [0 for i in task_idxs]
    self.task_high = [
        [.63, .55, .80, 1, .51, 1, .85, .89, .96, .85][i] for i in task_idxs
    ]

    # Load pretrained models
    self.ref_models = []
    tmp_opt = copy.deepcopy(opt)
    tmp_opt.imagenet_pretrained = 0

    for i, task_idx in enumerate(task_idxs):
      r = RefNet(tmp_opt, ds, sub_task_choice=[i])
      pretrained = torch.load(
          '%s/finetuned/%d/snapshot_model' % (opt.exp_root_dir, task_idx))
      r.load_state_dict(pretrained['model'], strict=False)
      r.cuda()
      r.eval()
      for p in r.parameters():
        p.requires_grad = False
      self.ref_models += [r]

    # If finetuning fc layers, copy fc weights from reference
    if opt.finetune_fc:
      for i in range(self.num_tasks):
        fc_name = 'out_%s' % self.task_names[i]
        fc_ref = self.ref_models[i]._modules[fc_name]
        self._modules[fc_name].load_state_dict(fc_ref.state_dict())

  def forward(self, x, task_idx, split, global_step):
    begin_idx = self.last_layer_idx
    end_idx = begin_idx + self.num_distill

    # Initial pass through task-specific resnet
    x = self.ref_models[task_idx](x)
    ref_feats = [x]
    matched_feats = [x]

    # Pass through teacher and shared layers
    ref_layer = self.ref_models[task_idx].resnet.layer3
    ref_end_bn = self.ref_models[task_idx].resnet.end_bns[0]
    shared_layer = self.resnet.layer3
    shared_end_bn = self.resnet.end_bns[task_idx]

    for l_idx in range(begin_idx, end_idx):
      ref_feats += [ref_layer[l_idx](ref_feats[-1], 0)]
      matched_feats += [shared_layer[l_idx](matched_feats[-1], task_idx)]
      if l_idx == 3:
        ref_feats[-1] = ref_end_bn(ref_feats[-1])
        matched_feats[-1] = shared_end_bn(matched_feats[-1])

    # Run through rest of pre-trained network
    x = matched_feats[-1]
    for l_idx in range(end_idx, 4):
      x = ref_layer[l_idx](x, 0)
    if end_idx < 4:
      x = ref_end_bn(x)

    x = self.resnet.avgpool(x)
    x = x.view(x.size(0), -1)

    # Final fully connected layer
    fc_name = 'out_%s' % self.task_names[task_idx]
    if not self.finetune_fc:
      final_fc = self.ref_models[task_idx]._modules[fc_name]
    else:
      final_fc = self._modules[fc_name]

    x = final_fc(x)

    ref_feats = torch.cat(ref_feats[1:], 1)
    matched_feats = torch.cat(matched_feats[1:], 1)

    return ref_feats, matched_feats, x


def initialize(*args, **kargs):
  return Net(*args, **kargs)

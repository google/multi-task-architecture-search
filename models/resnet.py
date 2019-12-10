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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def setup_extra_args(parser):
  parser.add_argument('--imagenet_pretrained', type=int, default=1)
  parser.add_argument('--separate_bn', type=int, default=1)
  parser.add_argument('--num_blocks', type=int, default=4)
  parser.add_argument('--bottleneck_ratio', type=float, default=1.)
  parser.add_argument('--network_size_factor', type=float, default=1.)
  parser.add_argument('--last_third_only', type=int, default=0)


class ConvBN(nn.Module):
  """Combined convolution/batchnorm, with support for separate normalization.

  Using a wrapper for the convolution operation makes it easier to
  add task-conditioned auxiliary functions to augment intermediate
  activations as we will be doing with the masks. This will make it
  easier to load and exchange weights across models whether or not
  they were trained with masks.
  """

  def __init__(self, in_channels, out_channels, stride=1, num_tasks=1):
    super(ConvBN, self).__init__()

    self.conv_aux = nn.Conv2d(
        in_channels, out_channels, 3, stride, padding=1, bias=False)
    self.conv = self.identity_fn(self.conv_aux)
    self.bns = nn.ModuleList(
        [nn.BatchNorm2d(out_channels) for i in range(num_tasks)])

  def identity_fn(self, cnv):
    return lambda x, y: cnv(x)

  def forward(self, x, task_idx=0):
    return self.bns[task_idx](self.conv(x, task_idx))


class ResBlock(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               stride=1,
               shortcut=0,
               num_tasks=1,
               bottleneck=1):
    super(ResBlock, self).__init__()

    f = int(out_channels * bottleneck)
    self.conv1 = ConvBN(in_channels, f, stride, num_tasks)
    self.conv2 = ConvBN(f, out_channels, 1, num_tasks)

    self.shortcut = shortcut
    if shortcut:
      self.avgpool = nn.AvgPool2d(2)

  def forward(self, x, task_idx=0):
    y = self.conv1(x, task_idx)
    y = self.conv2(F.relu(y, inplace=True), task_idx)

    if self.shortcut:
      x = self.avgpool(x)
      x = torch.cat((x, x * 0), 1)

    return F.relu(x + y)


class ResNet(nn.Module):

  def __init__(self, size_factor=1, num_blocks=4, num_tasks=1, bottleneck=1):
    super(ResNet, self).__init__()
    f = [int(size_factor * 2**(i + 5)) for i in range(4)]

    self.num_feat_ref = f
    self.num_tasks = num_tasks
    self.num_blocks = num_blocks
    self.pre_layers_conv = ConvBN(3, f[0], 1, num_tasks)

    for i in range(1, 4):
      tmp_bottleneck = 1 if i < 3 else bottleneck
      layers = [ResBlock(f[i - 1], f[i], 2, 1, num_tasks, tmp_bottleneck)]
      for j in range(1, num_blocks):
        layers += [ResBlock(f[i], f[i], 1, 0, num_tasks, tmp_bottleneck)]

      self.add_module('layer%d' % i, nn.Sequential(*layers))

    self.end_bns = nn.ModuleList([
        nn.Sequential(nn.BatchNorm2d(f[-1]), nn.ReLU(True))
        for i in range(num_tasks)
    ])
    self.avgpool = nn.AdaptiveAvgPool2d(1)

    # Weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def get_layer_ref(self):
    layer_ref = []
    for k in self.state_dict().keys():
      if 'conv' in k and 'weight' in k and 'bns' not in k:
        layer_ref += [k.split('.')[:-1]]

    return layer_ref

  def get_module(self, l):
    tmp_m = self
    if len(l) > 1:
      tmp_m = tmp_m._modules[l[0]]
    if len(l) > 2:
      tmp_m = tmp_m[int(l[1])]
    if len(l) > 3:
      tmp_m = tmp_m._modules[l[2]]
    if len(l) > 4:
      tmp_m = tmp_m[int(l[3])]

    return tmp_m

  def forward(self, x, task_idx):
    x = self.pre_layers_conv(x, task_idx)
    for l in [self.layer1, self.layer2, self.layer3]:
      for m in l:
        x = m(x, task_idx)

    x = self.end_bns[task_idx](x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    return x


class Net(nn.Module):

  def __init__(self, opt, ds, metaparams=None, masks=None,
               sub_task_choice=None):
    super(Net, self).__init__()
    self.dropout = opt.dropout
    self.separate_bn = opt.separate_bn

    if sub_task_choice is None:
      sub_task_choice = [i for i in range(len(ds['train']))]
    self.num_tasks = len(sub_task_choice)
    self.num_out = [ds['train'][i].num_out for i in sub_task_choice]
    self.task_names = [ds['train'][i].task_name for i in sub_task_choice]

    # Initialize ResNet
    self.resnet = ResNet(opt.network_size_factor, opt.num_blocks,
                         self.num_tasks if opt.separate_bn else 1,
                         opt.bottleneck_ratio)

    # Final fully connected layers
    f = int(256 * opt.network_size_factor)
    self.final_n_feat = f
    for t in range(self.num_tasks):
      self.add_module('out_%s' % self.task_names[t],
                      nn.Linear(f, self.num_out[t]))

    if opt.imagenet_pretrained:
      print('Loading pretrained Imagenet model.')
      imgnet_path = '%s/finetuned/0/snapshot_model' % opt.exp_root_dir
      pretrained = torch.load(imgnet_path)
      if opt.bottleneck_ratio != 1:
        pretrained['model'] = {
            k: v for k, v in pretrained['model'].items() if 'layer3' not in k
        }
      self.load_state_dict(pretrained['model'], strict=False)

      # Copy batchnorm weights in all layers
      for l in self.resnet.get_layer_ref():
        tmp_m = self.resnet.get_module(l)
        if len(tmp_m.bns) > 1:
          for bn_idx in range(1, len(tmp_m.bns)):
            tmp_m.bns[bn_idx].load_state_dict(tmp_m.bns[0].state_dict())

    if opt.last_third_only:
      # Do not update conv weights of first two-thirds of model (still update BN)
      layer_names = ['pre_layers_conv', 'layer1', 'layer2']

      resnet_params = []
      for l_name in layer_names:
        l = self.resnet._modules[l_name]
        for k in l.state_dict():
          if 'conv_aux' in k:
            tmp_m = self.resnet.get_module([l_name] + k.split('.')[:-1])
            tmp_m.conv_aux.weight.requires_grad = False
        resnet_params += [p for p in l.parameters() if len(p.shape) == 1]

      resnet_params += [p for p in self.resnet.layer3.parameters()]

    else:
      # Train the full model
      resnet_params = [p for p in self.resnet.parameters()]

    self.net_parameters = []
    for t in range(self.num_tasks):
      task_out = self._modules['out_%s' % self.task_names[t]]
      task_params = [p for p in task_out.parameters()]
      self.net_parameters += [resnet_params + task_params]

  def forward(self, x, task_idx, split, global_step):
    x = self.resnet(x, task_idx if self.separate_bn else 0)
    if split == 'train' and self.dropout:
      x = F.dropout(x, self.dropout)
    x = self._modules['out_%s' % self.task_names[task_idx]](x)

    return x


def initialize(*args, **kargs):
  return Net(*args, **kargs)

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

"""Class to manage metaparameters."""

from copy import deepcopy

from mtl.util import calc
import numpy as np
import torch
import torch.nn


class ParamManager(torch.nn.Parameter):

  def __new__(cls, shape=None):
    data = None if shape is None else torch.Tensor(*shape)
    return super().__new__(cls, data=data, requires_grad=False)

  def __init__(self):
    super().__init__()
    self.is_continuous = True
    self.valid_range = None

  def get_default(self):
    return None

  def random_sample(self):
    # Sample within defined range with appropriate scaling
    vals = torch.rand(self.shape)
    vals = calc.map_val(vals, *self.valid_range)
    # Discretize if necessesary
    if not self.is_continuous:
      vals = vals.astype(int)

    return vals

  def set_to_default(self):
    self.data[:] = self.get_default()

  def set_to_random(self):
    self.data[:] = self.random_sample()

  def mutate(self, delta=.1):
    return None

  def copy(self):
    return deepcopy(self)


class Metaparam(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.is_command = False
    self.is_parameterized = None
    self.model = None

  def parameters(self, keys=None):
    for name, param in self.named_parameters():
      tmp_name = name.split('.')[0]
      if keys is None or tmp_name in keys:
        yield param

  def get_params(self, keys):
    return torch.nn.utils.parameters_to_vector(self.parameters(keys)).detach()

  def update_params(self, keys, data):
    torch.nn.utils.vector_to_parameters(data, self.parameters(keys))

  def copy_from(self, src, keys):
    self.update_params(keys, src.get_params(keys))

  def parameterize(self, model, search, inp_size):
    # Loop through all keys, get data shape size
    data_ref = [self._parameters[k].data for k in search]
    dim_ref = [d.shape for d in data_ref]
    # Determine output size (flattened/concatted data)
    out_size = int(sum([np.prod(d) for d in dim_ref]))
    # Initialize model
    self.model = model(inp_size, out_size)
    self.is_parameterized = search
    self.is_command = False

  def reparameterize(self, x):
    # Update metaparameters after forward call of model
    new_params = self.model(x)
    self.update_params(self.is_parameterized, new_params)

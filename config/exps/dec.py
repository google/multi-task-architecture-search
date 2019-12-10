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

option_defaults = {
  'dataset': 'decathlon',
  'model': 'masked_resnet',
  'task_choice': '1-2-3-4-5-6-7-8-9',
  'param_init': 'random',
  'weight_decay': 1e-4,
  'drop_rate': .3,
  'num_rounds': 40,
  'train_iters': 5000,
  'drop_lr_iters': '150000',
  'num_unique': 1,
  'fixed_seed': 0,
  'last_third_only': 1,
  'imagenet_pretrained': 1,
}

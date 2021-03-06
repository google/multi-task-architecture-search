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

is_meta = True

option_defaults = {
  'metaoptimizer': 'runner',
  'worker_mode': 'cmd',
  'num_procs': 8,
  'gpu_choice': '0,1,2,3',
  'param': '',
  'num_samples': 3,
  'distribute': 1,
  'cleanup_experiment': 1,
}

base_cmd = '--config exps.dist --suppress_output 1'.split(' ')
exp_list = []
exp_names = []

for i in range(3900):
  exp_list += ['--metaparam_load all_metaparams/%d/%d' % (i // 100, i % 100)]
  exp_names += ['m%d/%d' % (i // 100, i % 100)]
exp_list = [e.split(' ') for e in exp_list]

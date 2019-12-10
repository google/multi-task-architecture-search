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
  'num_procs': 4,
  'gpu_choice': '0,1,2,3',
  'param': '',
  'num_samples': 3,
  'distribute': 1,
}

base_cmd = '--config exps.finetune --suppress_output 1'.split(' ')
exp_list = []
exp_names = []

tmp_exps = ['', '--last_third_only 0',
            '--imagenet_pretrained 0 --last_third_only 0 --num_rounds 30 --drop_lr_iters 100000']
tmp_names = ['', 'full', 'scratch']
for task_idx in range(1,10):
  for exp_idx, exp_type in enumerate(tmp_exps):
    exp_list += ['--task_choice %d %s' % (task_idx, exp_type)]
    exp_names += ['t%d%s' % (task_idx, tmp_names[exp_idx])]
exp_list = [e.split(' ') for e in exp_list]

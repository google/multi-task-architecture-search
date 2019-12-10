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
  'metaoptimizer': 'es',
  'worker_mode': 'cmd',
  'num_procs': 8,
  'gpu_choice': '0,1,2,3',
  'distribute': 1,
  'param': 'partition',
  'search': 'partition',
  'num_samples': 10000,
  'task_choice': '1-2-3-4-5-6-7-8-9',
  'init_random': 32,
  'learning_rate': .25,
  'momentum': .8,
  'delta_size': .04,
  'num_deltas': 8,
  'num_to_use': 7,
  'num_unique': 1,
  'cleanup_experiment': 1,
  'do_weight_reg': 2,
  'diag_weight_decay': .001,
  'multiobjective': 0,
}

base_cmd = '--config exps.dist --suppress_output 1'.split(' ')

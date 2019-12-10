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
  'worker_mode': 'debug',
  'num_procs': 1,
  'param': 'partition',
  'search': 'partition',
  'num_samples': 10000,
  'task_choice': '0-1-2-3-4-5-6-7-8-9',
  'learning_rate': 1,
  'momentum': .8,
  'delta_size': .005,
  'num_deltas': 32,
  'num_to_use': 24,
}

base_cmd = None

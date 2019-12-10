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

"""Command line options for meta optimization."""


def setup_args(parser):
  parser.add_argument('--metaoptimizer', type=str, default='es')
  parser.add_argument('-n', '--num_samples', type=int, default=10000)
  parser.add_argument('--worker_mode', type=str, default='cmd')
  parser.add_argument('--num_procs', type=int, default=4)
  parser.add_argument('--distribute', type=int, default=1)
  parser.add_argument('--cleanup_experiment', type=int, default=0)

  # Maximize a reward or minimize a loss?
  parser.add_argument('--maximize', type=int, default=1)
  parser.add_argument('--worst_score', type=float, default=0)

  # Parameter options
  parser.add_argument('-p', '--param', type=str, default='partition')
  parser.add_argument('-s', '--search', type=str, default='partition')
  parser.add_argument('--cmd_config', type=str, default='')

  # Meta optimization debugging options
  parser.add_argument('--meta_eval_noise', type=float, default=0.)

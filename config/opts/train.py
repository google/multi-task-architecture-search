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

"""Command line options for network training."""


def setup_args(parser):
  # Dataset options
  parser.add_argument('-d', '--dataset', type=str, default='decathlon')
  parser.add_argument('--num_data_threads', type=int, default=1)
  parser.add_argument('--train_on_valid', type=int, default=0)
  parser.add_argument('--validate_on_train', type=int, default=0)
  parser.add_argument('--use_test', type=int, default=0)

  # Training length options
  parser.add_argument('--num_rounds', type=int, default=100)
  parser.add_argument('--train_iters', type=int, default=4000)
  parser.add_argument('--early_stop_thr', type=float, default=.0)
  parser.add_argument('--curriculum', type=str, default='train_accuracy')

  # Training hyperparameters
  parser.add_argument('--optimizer', type=str, default='SGD')
  parser.add_argument('-l', '--learning_rate', type=float, default=.05)
  parser.add_argument('--batchsize', type=int, default=64)
  parser.add_argument('--valid_batchsize', type=int, default=0)
  parser.add_argument('--momentum', type=float, default=.9)
  parser.add_argument('--weight_decay', type=float, default=1e-4)
  parser.add_argument('--clip_grad', type=float, default=0.)
  parser.add_argument('--dropout', type=float, default=0.3)
  parser.add_argument('--temperature', type=float, default=0.3)
  parser.add_argument('--curriculum_bias', type=float, default=0.15)

  parser.add_argument('--drop_lr_iters', type=str, default='')
  parser.add_argument('--drop_lr_factor', type=int, default=10)

  # Task options
  parser.add_argument('-t', '--task', type=str, default='multiclass')
  parser.add_argument('--metaparam', type=str, default='')
  parser.add_argument('--metaparam_load', type=str, default='')

  # Model
  parser.add_argument('-m', '--model', type=str, default='masked_resnet')
  parser.add_argument('--pretrained', type=str, default='')

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

"""Prepare json file for Decathlon submission server."""
import json

from mtl.config.opts import parser
from mtl.util.datasets import decathlon
import torch


opt = parser.parse_command_line()
ds, _ = decathlon.initialize(opt)
ds = ds['valid']

# Load predictions
imgnet_p = torch.load(opt.exp_root_dir +
                      '/imgnet_test/final_predictions')['preds']
p = torch.load(opt.exp_dir + '/final_predictions')['preds']
all_preds = imgnet_p + p

tmp_result = []
for task_idx in range(10):
  n_ims = len(ds[task_idx])
  cat_ref = ds[task_idx].annot['cat_id_ref']
  im_ids = ds[task_idx].annot['img_id']
  for im_idx in range(n_ims):
    tmp_result += [{
        'image_id': int(im_ids[im_idx]),
        'category_id': int(cat_ref[all_preds[task_idx][im_idx]])
    }]

with open(opt.exp_dir + '/test_submission.json', 'w') as f:
  json.dump(tmp_result, f)

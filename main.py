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

"""Main file for running experiments..
"""
import importlib
import os
import subprocess

from mtl.config.opts import parser
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


def train(opt):
  """Run standard network training loop.

  Args:
    opt: All experiment and training options (see mtl/config/opts).
  """

  if opt.fixed_seed:
    print('Fixing random seed')
    np.random.seed(9999)
    torch.manual_seed(9999)
    torch.cuda.manual_seed(9999)
    torch.backends.cudnn.deterministic = True

  ds = importlib.import_module('mtl.util.datasets.' + opt.dataset)
  ds, dataloaders = ds.initialize(opt)

  task = importlib.import_module('mtl.train.' + opt.task)
  sess = task.Task(opt, ds, dataloaders)
  sess.cuda()

  splits = [s for s in ['train', 'valid'] if opt.iters[s] > 0]
  start_round = opt.last_round - opt.num_rounds

  # Main training loop
  for round_idx in range(start_round, opt.last_round):

    sess.valid_accuracy_track = [[] for _ in range(sess.num_tasks)]
    for split in splits:

      print('Round %d: %s' % (round_idx, split))
      train_flag = split == 'train'
      sess.set_train_mode(train_flag)

      if split == 'valid':
        sess.prediction_ref = [{} for _ in range(sess.num_tasks)]

      for step in tqdm(range(opt.iters[split]), ascii=True):
        global_step = step + round_idx * opt.iters[split]
        sess.run(split, global_step)
        if train_flag: sess.update_weights()

        if (split == 'train' and opt.drop_learning_rate
            and global_step in opt.drop_learning_rate):
          opt.learning_rate /= opt.drop_lr_factor
          print('Dropping learning rate to %.2f' % opt.learning_rate)
          for opt_key in sess.checkpoint_ref['optim']:
            for p in sess.__dict__[opt_key].param_groups:
              p['lr'] = opt.learning_rate

        # Update Tensorboard
        if global_step % 500 == 0 or (split == 'valid' and global_step % 50):
          for i in range(len(ds[split])):
            sess.tb.update(ds[split][i].task_name, split, global_step,
                           sess.get_log_vals(split, i))

    torch.save({'preds': sess.prediction_ref},
               '%s/final_predictions' % opt.exp_dir)

    # Update accuracy history
    sess.score = np.array([np.array(a).mean()
                           for a in sess.valid_accuracy_track]).mean()
    print('Score:', sess.score)

    for i in range(sess.num_tasks):
      for s in splits:
        if s == 'valid':
          tmp_acc = np.array(sess.valid_accuracy_track[i]).mean()
          sess.log['accuracy'][i][s] = tmp_acc
        sess.log['accuracy_history'][i][s] += [sess.log['accuracy'][i][s]]

    sess.save(opt.exp_dir + '/snapshot')
    with open(opt.exp_dir + '/last_round', 'w') as f:
      f.write('%d\n' % (round_idx + 1))

    if (opt.iters['valid'] > 0 and sess.score < opt.early_stop_thr):
      break


def worker(opt, p_idx, cmd_queue, result_queue, debug_param):
  """Worker thread for managing parallel experiment runs.

  Args:
    opt: Experiment options
    p_idx: Process index
    cmd_queue: Queue holding experiment commands to run
    result_queue: Queue to submit experiment results
    debug_param: Shared target for debugging meta-optimization
  """
  gpus = list(map(int, opt.gpu_choice.split(',')))
  gpu_choice = gpus[p_idx % len(gpus)]
  np.random.seed()

  try:
    while True:
      msg = cmd_queue.get()
      if msg == 'DONE': break
      exp_count, mode, cmd, extra_args = msg

      if mode == 'debug':
        # Basic operation for debugging/sanity checking optimizers
        exp_id, param = cmd
        pred_param = param['partition'][0]

        triu_ = torch.Tensor(np.triu(np.ones(debug_param.shape)))
        score = -np.linalg.norm((debug_param - pred_param)*triu_)
        score += np.random.randn() * opt.meta_eval_noise

        tmp_acc = {'accuracy': [{'valid': score} for i in range(10)]}
        result = {'score': score, 'log': tmp_acc}

      elif mode == 'cmd':
        # Run a specified command
        tmp_cmd = cmd
        if opt.distribute:
          tmp_cmd += ['--gpu_choice', str(gpu_choice)]
        tmp_cmd += extra_args
        exp_id = tmp_cmd[tmp_cmd.index('-e') + 1]

        print('%d:' % p_idx, ' '.join(tmp_cmd))
        subprocess.call(tmp_cmd)

        # Collect result
        log_path = '%s/%s/snapshot_extra' % (opt.exp_root_dir, exp_id)
        try:
          result = torch.load(log_path)
        except Exception as e:
          print('Error loading result:', repr(e))
          result = None

        if opt.cleanup_experiment:
          # Remove extraneous files that take up disk space
          exp_dir = '%s/%s' % (opt.exp_root_dir, exp_id)
          cleanup_paths = [exp_dir + '/snapshot_optim',
                           exp_dir + '/snapshot_model']
          dir_files = os.listdir(exp_dir)
          tfevent_files = ['%s/%s' % (exp_dir, fn)
                           for fn in dir_files if 'events' in fn]
          cleanup_paths += tfevent_files

          for cleanup in cleanup_paths:
            subprocess.call(['rm', cleanup])

      result_queue.put([exp_id, result, exp_count])

  except KeyboardInterrupt:
    print('Keyboard interrupt in process %d' % p_idx)
  finally:
    print('Exiting process %d' % p_idx)


def main():
  # Parse command line options
  opt = parser.parse_command_line()
  # Set GPU
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_choice

  if opt.is_meta:
    # Initialize queues
    cmd_queue = mp.Queue()
    result_queue = mp.Queue()

    # Set up target debug params
    debug_param = torch.rand(2, 10, 10)

    # Start workers
    workers = []
    for i in range(opt.num_procs):
      worker_args = (opt, i, cmd_queue, result_queue, debug_param)
      worker_p = mp.Process(target=worker, args=worker_args)
      worker_p.daemon = True
      worker_p.start()
      workers += [worker_p]

    # Initialize and run meta optimizer
    metaoptim = importlib.import_module('mtl.meta.optim.' + opt.metaoptimizer)
    metaoptim = metaoptim.MetaOptimizer(opt)
    metaoptim.run(cmd_queue, result_queue)

    # Clean up workers
    for i in range(opt.num_procs):
      cmd_queue.put('DONE')
    for worker_p in workers:
      worker_p.join()

  else:
    # Run standard network training
    train(opt)

if __name__ == '__main__':
  main()

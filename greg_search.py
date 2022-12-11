import numpy as np
import os
from multiprocessing import Pool
import math
import sys
import subprocess
import json
import shutil

log = open(f'{sys.argv[0].replace(".py", "")}.log', 'w')

m = np.max(np.round(np.linspace(-0.01, 0.2, 12), 2))
weight_image_contexts = sorted(np.round(np.linspace(-0.01, 0.2, 12), 2), key=lambda x: x!=m)
weight_context_glosss = sorted(np.round(np.linspace(-0.01, 0.2, 12), 2), key=lambda x: x!=m)
weight_image_glosss = sorted(np.round(np.linspace(-0.01, 0.2, 12), 2), key=lambda x: x!=m)
pool_funcs = ['max', 'mean'][:1]
cache = {pf: [] for pf in pool_funcs}

trial_cmd = 'python greg.py -d semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/ --model openai/clip-vit-base-patch32'
k = 500
sample_cmd = f'python greg.py -d semeval-2023-task-1-V-WSD-train-v1/sample/data.{k}.txt -g semeval-2023-task-1-V-WSD-train-v1/sample/gold.{k}.txt -i semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ --model openai/clip-vit-base-patch32'
train_cmd = 'python greg.py -d semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ --model openai/clip-vit-base-patch32'

base_cmd = sample_cmd

# good = 0
# bad = 0
# cmds = []
# hyps = []
# done_eq = {'mean': False, 'max': False}
# for w_ic in weight_image_contexts:
#   for w_cg in weight_context_glosss:
#     for w_ig in weight_image_glosss:
#       for pf in pool_funcs:
#         cmd = f'-w_ic {w_ic} -w_cg {w_cg} -w_ig {w_ig} -pf {pf}'
#         arr = np.array([w_ic, w_cg, w_ig])
#         ratio = list(arr / arr.sum())
#         eq = w_ic == w_cg and w_cg == w_ig
#         if (eq and done_eq[pf]) or (ratio in cache[pf]): 
#           print(cmd, 'bad')
#           bad += 1
#         else:
#           if eq and not done_eq[pf]:
#             done_eq[pf] = True
#           print(cmd, 'good')
#           cmds.append(f'{base_cmd} {cmd}')
#           hyps.append(cmd)
#           good += 1
#           cache[pf].append(ratio)

# print(f'Good: {good}, Bad: {bad}, Difference: {good - bad}')
# print(weight_image_contexts)
# print(weight_context_glosss)
# print(weight_image_glosss)

cmds = []
hyps = []
for w_ic, w_cg, w_ig in {(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1)}:
  cmd = f'-w_ic {w_ic} -w_cg {w_cg} -w_ig {w_ig} -pf max'
  cmds.append(f'{base_cmd} {cmd}')
  hyps.append(cmd)

shutil.rmtree('logs.out', ignore_errors=True)

output_dict = {}

def execute_cmd(tup):
  gpu_dev_env_var, cmd, hyp = tup
  print(f'Executing: {gpu_dev_env_var} {cmd}')
  output = subprocess.check_output([f'{gpu_dev_env_var} {cmd}'], shell=True).decode('utf-8')
  splits = output.split('\n')
  print(f'Output: {output}')
  for s in splits:
    if s.startswith('Accuracy: '):
      acc = float(s.split(': ')[1].strip())
      output_dict[hyp] = acc
      log.write(f'{hyp}: {acc}\n')
      log.flush()
      return acc
  return -1.

with Pool(10) as pool:
  devs = [f'CUDA_VISIBLE_DEVICES={c}' for c in [0, 1]] * math.ceil(len(cmds) / 2)
  results = pool.imap_unordered(execute_cmd, zip(devs, cmds, hyps))
  for i, result in enumerate(results):
    print(hyps[i], result)
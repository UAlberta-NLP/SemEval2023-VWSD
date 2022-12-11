import random
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-id', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
parser.add_argument('-ig', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
parser.add_argument('-o', default='semeval-2023-task-1-V-WSD-train-v1/sample')
parser.add_argument('-s', default=42, type=int)
parser.add_argument('-k', default=500, type=int)
args = parser.parse_args()

os.makedirs(args.o, exist_ok=True)
data = open(args.id).readlines()
gold = open(args.ig).readlines()
data_out = open(os.path.join(args.o, f'data.{args.k}.txt'), 'w')
gold_out = open(os.path.join(args.o, f'gold.{args.k}.txt'), 'w')

random.seed(args.s)
lines = random.sample(range(len(data)), k=args.k)
for l in tqdm(lines):
  d = data[l]
  g = gold[l]
  data_out.write(d)
  gold_out.write(g)
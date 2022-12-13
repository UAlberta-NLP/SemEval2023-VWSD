# nohup python text_image_baseline.py -d semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt  -g semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt  -i  semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ &
# nohup python text_image_baseline.py -d semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ --model openai/clip-vit-large-patch14 > 14.out &

import argparse
import glob
import os
from time import time
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import termcolor
import torch
from tqdm import tqdm
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000

import sys
sys.path.append('.')
from utils import cos_sim

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='data/trial.data.txt')
parser.add_argument('--gold', '-g', default='data/trial.gold.txt')
parser.add_argument('--image-dir', '-i', default='data/all_images')
parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
parser.add_argument('--instance_batch_size', '-ibs', default=10, type=int, help='This does not follow the conventional meaning of batch size. Kindly take note.')
parser.add_argument('--output', '-o', default=None)
args = parser.parse_args()

if args.output is None:
  args.output = f"{args.model.replace('/', '_')}_log.out"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
assert len(data) == len(gold_data)
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))
correct, total = 0, 0

out = open(args.output, 'w')
ibs = args.instance_batch_size
i = 0
j = i + ibs
inst_size = 10 # TODO: softcode
for i in tqdm(range(0, len(data), ibs), 'Processing images and text...'):
  if i >= len(data):
    break

  instance = data[i:j]
  gold = gold_data[i:j]

  words, contexts, image_pathss = [], [], []
  for inst in instance:
    word, context, *image_paths = inst
    words.append(word)
    contexts.append(context)
    image_pathss.extend(image_paths)

  # print(words, contexts, image_pathss)
  images = [Image.open(os.path.join(args.image_dir, i)) for i in image_pathss]
  inputs = processor(text=contexts, images=images, return_tensors="pt", padding=True).to(device)
  outputs = model(**inputs)
  
  for k in range(len(instance)):
    img_e = outputs.image_embeds[k*inst_size:(k+1)*inst_size]
    txt_e = outputs.text_embeds[k:k+1]
    txt_e = (txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)).T

    sim = cos_sim(img_e, txt_e)
    word = words[k]
    image_paths = image_pathss[k*inst_size:(k+1)*inst_size]
    g_k = gold[k]
    best = image_paths[sim.argmax()]
    total += 1
    is_correct = int(best == g_k)
    correct += 1 if is_correct else 0
    color = termcolor.colored('right', 'green') if is_correct else termcolor.colored('wrong', 'red')
    out.write(f'{word} {best} {g_k} {image_paths} -> {"right" if is_correct else "wrong"}\n')
  
  out.flush()
  i += ibs
  j += ibs

msg = f'\nAccuracy: {correct / total}'
out.write(msg)
print(msg)
out.close()

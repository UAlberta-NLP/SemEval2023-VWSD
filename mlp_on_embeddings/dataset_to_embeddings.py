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
parser.add_argument('--instance_batch_size', '-ibs', default=1, type=int, help='This does not follow the conventional meaning of batch size.')
parser.add_argument('--output', '-o', default=None)
args = parser.parse_args()

if args.output is None:
  args.output = f"{args.model.replace('/', '_')}.pt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
assert len(data) == len(gold_data)
correct, total = 0, 0

all_image_names = []
for instance in data:
  word, context, *image_paths = instance
  all_image_names.extend(image_paths)

print(f'All images: {len(all_image_names):,d}')
all_image_names = set(all_image_names)
print(f'Unique images: {len(all_image_names):,d}')

txt_dict = {}
img_dict = {}
out = open(args.output, 'w')
ibs = args.instance_batch_size
i = 0
j = i + ibs
inst_size = 10

with torch.no_grad():
  for i in tqdm(range(0, len(data), ibs), 'Processing images and text...'):
    if i >= len(data):
      break
    j = i + ibs

    instance = data[i:j]
    gold = gold_data[i:j]

    words, contexts, image_pathss = [], [], []
    for inst in instance:
      word, context, *image_paths = inst
      words.append(word)
      contexts.append(context)
      image_pathss.extend(image_paths)
    
    if len(contexts) > 0:
      txt_inputs = processor(text=contexts, return_tensors="pt", padding=True).to(device)
      txt_e = model.get_text_features(**txt_inputs)
      txt_e = (txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)).to('cpu')
      local_txt_dict = {c:e for c, e in zip(contexts, txt_e)}
      txt_dict.update(local_txt_dict)

    image_pathss = [i for i in image_pathss if i not in img_dict]
    if len(image_pathss) > 0:
      images = [Image.open(os.path.join(args.image_dir, i)) for i in image_pathss]
      img_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
      img_e = model.get_image_features(**img_inputs)
      img_e = (img_e / img_e.norm(p=2, dim=-1, keepdim=True)).to('cpu')
      local_img_dict = {p:e for p, e in zip(image_pathss, img_e)}
      img_dict.update(local_img_dict)

torch.save({'txt': txt_dict, 'img': img_dict}, args.output)
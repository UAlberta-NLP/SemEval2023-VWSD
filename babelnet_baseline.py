# Start docker service
# docker run -d --name babelnet -v /local/storage/babelnet5/BabelNet-5.0/ -p 7780:8000 -p 7790:1234 babelscape/babelnet-rpc:latest

# Run java pre-program
# sh run-bgwi.sh /home/ogezi/ideas/v-wsd/data/trial.data.txt /home/ogezi/ideas/v-wsd/data/images.json

import argparse
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
import glob
import os
import json
from PIL import Image
from utils import cos_sim
import termcolor

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--bn-image-meta', default='data/images.json')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--bn-image-dir', default='data/bn_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

meta = json.load(open(args.bn_image_meta))
data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold = [l.strip() for l in open(args.gold).readlines()]
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))

bn_mean_latents = {}
for word, senses in meta.items():
  if word not in bn_mean_latents:
    bn_mean_latents[word] = {}
  for sense in senses:
    id = sense['id']
    target_files = glob.glob(os.path.join(args.bn_image_dir, word, id, '*'))
    if len(target_files) == 0:
      bn_mean_latents[word][id] = torch.zeros(512) + 1e-9
      continue
    images = [Image.open(i) for i in target_files]
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    image_outputs = model.get_image_features(**image_inputs)
    bn_mean_latents[word][id] = image_outputs.mean(dim=0)
    print(word, id)

correct, total = 0, 0
for instance, gold in zip(data, gold):
  word, context, *image_paths = instance
  images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
  image_outputs = model.get_image_features(**image_inputs)
  mx_idx = 0
  mx = 0
  for idx, (id, mean_latents) in enumerate(bn_mean_latents[word].items()):
    mean_latents = mean_latents.to(device)
    sim = cos_sim(image_outputs, mean_latents)
    # print(image_outputs, mean_latents)
    # print(sim.shape, sim)
    mx_idx = sim.argmax() if mx < sim.max() else mx_idx
    mx = sim.max() if mx < sim.max() else mx
  best = image_paths[mx_idx]
  total += 1
  correct += 1 if best == gold else 0
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, best, gold, '->', color)

print(f'\nAccuracy: {correct / total}')
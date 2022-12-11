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
parser.add_argument('--bn-image-meta', default='data/bn_images.json')
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

bn_latents = {}
bn_maps = {}
eps = 1e-9
w = 'swing'
with torch.no_grad():
  for word, senses in meta.items():
    cnt = 0
    # if word != w:
    #   continue
    if word not in bn_latents:
      bn_latents[word] = {}
      bn_maps[word] = []
    for sense in senses:
      id = sense['id']
      # print(word, id)
      target_files = glob.glob(os.path.join(args.bn_image_dir, word, id, '*'))
      for t in target_files:
        bn_maps[word].append(t)
        # print(cnt, t)
        cnt += 1
      if len(target_files) == 0:
        # bn_latents[word][id] = torch.zeros((1, 512), device=device) + eps
        continue
      images = [Image.open(i) for i in target_files]
      image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
      image_outputs = model.get_image_features(**image_inputs)
      bn_latents[word][id] = image_outputs

correct, total = 0, 0
thresh = 1. - (1e-6)
data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold = [l.strip() for l in open(args.gold).readlines()]
with torch.no_grad():
  for instance, gold in zip(data, gold):
    word, context, *image_paths = instance
    # if word != w:
    #   continue
    images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    image_outputs = model.get_image_features(**image_inputs)
    all_word_latents = torch.cat([i.to(device) for i in bn_latents[word].values()], dim=0)
    # print(all_word_latents.shape)
    latents = all_word_latents.T
    sim_matrix = cos_sim(image_outputs, latents)
    # print(sim_matrix, sim_matrix.shape)
    argmax_2d = (sim_matrix == torch.max(sim_matrix)).nonzero()
    # print(sim_matrix)
    acceptable = torch.where(sim_matrix >= thresh, 1, 0)
    acceptable_candidate_idx = torch.max(acceptable, dim=0).values
    acceptable_candidates = [i for idx, i in enumerate(image_paths) if acceptable_candidate_idx[idx] == 1.]
    print(word, acceptable_candidates, acceptable_candidate_idx)
    
    if len(acceptable_candidates) > 0:
      images = [Image.open(os.path.join(args.image_dir, i)) for i in acceptable_candidates]
      image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
      image_outputs = model.get_image_features(**image_inputs)
    else:
      acceptable_candidates = image_paths
    
    txt_inputs = tokenizer(text=[context], padding=True, return_tensors="pt").to(device)
    txt_e = model.get_text_features(**txt_inputs).T
    sim = cos_sim(image_outputs, txt_e)
    best = acceptable_candidates[sim.argmax()]
    total += 1
    correct += 1 if best == gold else 0
    color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
    print(word, best, gold, '->', color)

print(f'\nAccuracy: {correct / total}')
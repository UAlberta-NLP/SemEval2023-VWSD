'''
  After using a text-WSD system to disambiguate the focus words in context, this program uses those synsets as labels for an image classifier.
'''

import argparse
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from utils import cos_sim
import termcolor
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser()
parser.add_argument('--sense-key-file', '-s', default='data/text-wsd/gold.txt')
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

def line_to_tuple(line):
  splits = line.strip().split('\t')
  return tuple(splits)

sense_keys = [l.strip().split(' ')[1] for l in open(args.sense_key_file).readlines()]
image_data = [line_to_tuple(l) for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]

assert len(sense_keys) == len(image_data)
assert len(image_data) == len(gold_data)

total = 0
correct = 0
for sk, dat, gold in zip(sense_keys, image_data, gold_data):
  word = dat[0]
  context = dat[1]
  image_paths = dat[2:]
  syn = wn.lemma_from_key(sk).synset()
  text = [syn.definition()]

  images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(device)
  outputs = model(**inputs)

  img_e = outputs.image_embeds
  txt_e = outputs.text_embeds
  txt_e = (txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)).T
  sim = cos_sim(img_e, txt_e)
  best = image_paths[sim.argmax()]
  total += 1
  correct += 1 if best == gold else 0
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, f'"{text}"', best, gold, '->', color)

print(f'\nAccuracy: {correct / total}')
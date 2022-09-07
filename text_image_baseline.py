import argparse
import glob
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import termcolor
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold = [l.strip() for l in open(args.gold).readlines()]
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))
correct, total = 0, 0
for instance, gold in zip(data, gold):
  word, context, *image_paths = instance
  images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  model.get_text_features()
  inputs = processor(text=[context], images=images, return_tensors="pt", padding=True).to(device)
  outputs = model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = logits_per_image.softmax(dim=0)
  best = image_paths[probs.argmax()]
  total += 1
  correct += 1 if best == gold else 0
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, best, gold, '->', color)

print(f'\nAccuracy: {correct / total}')
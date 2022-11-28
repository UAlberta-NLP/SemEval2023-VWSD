import argparse
import glob
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import termcolor
import torch
import sys
sys.path.append('.')
from utils import cos_sim

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CLIPModel.from_pretrained(args.model).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold = [l.strip() for l in open(args.gold).readlines()]
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))
correct, total = 0, 0

# for instance, gold in zip(data, gold):
#   word, context, *image_paths = instance
#   images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  
#   text_inputs = tokenizer(text=[context], padding=True, return_tensors="pt").to(device)
#   text_embeddings = model.get(**text_inputs).T
  
#   image_inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
  
#   image_embeddings = model.get_image_features(**image_inputs)
  
#   cosine_similarity = cos_sim(image_embeddings, text_embeddings)
  
#   best = image_paths[cosine_similarity.argmax()]
  
#   total += 1
#   correct += 1 if best == gold else 0
#   color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  
#   print(word, best, gold, '->', color)
#   print(text_inputs, text_embeddings.shape)
#   print(model.get_text_features(**text_inputs, return_dict=True, output_hidden_states=True))
#   break

# for instance, gold in zip(data, gold):
#   word, context, *image_paths = instance
#   images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
#   inputs = processor(text=[context], images=images, return_tensors="pt", padding=True).to(device)
#   outputs = model(**inputs)
#   logits_per_image = outputs.logits_per_image
#   probs = logits_per_image.softmax(dim=0)
#   best = image_paths[probs.argmax()]
#   total += 1
#   correct += 1 if best == gold else 0
#   color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
#   print(word, best, gold, '->', color)
#   print(outputs.text_model_output.last_hidden_state.shape,  outputs.text_model_output.pooler_output.shape)
#   break

for instance, gold in zip(data, gold):
  word, context, *image_paths = instance
  images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  inputs = processor(text=[context], images=images, return_tensors="pt", padding=True).to(device)
  outputs = model(**inputs)
  # logits_per_image = outputs.logits_per_image
  # probs = logits_per_image.softmax(dim=0)
  # img_e = outputs.vision_model_output.pooler_output
  # txt_e = outputs.text_model_output.pooler_output.T
  img_e = outputs.image_embeds
  # get hidden states for tokens from the second to the second to the last
  ctx_x = model.text_projection(outputs.text_model_output[0][:, 1:-1, :]).mean(dim=1)
  # get hidden states for the last token
  cls_x = model.text_projection(outputs.text_model_output[0][:, -1:, :]).squeeze(dim=1)
  # print(ctx_x.shape, cls_x.shape)
  # take a weighted average of states
  x = torch.cat((ctx_x * 0.75, cls_x * 0.25), dim=0).sum(dim=0)
  # y = torch.cat((ctx_x, cls_x), dim=0).mean(dim=0)
  # print(x == y)
  # print(x_comb.shape)
  txt_e = (x / x.norm(p=2, dim=-1, keepdim=True)).T
  sim = cos_sim(img_e, txt_e)
  best = image_paths[sim.argmax()]
  total += 1
  correct += 1 if best == gold else 0
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, best, gold, '->', color)
  # print(outputs.text_model_output.last_hidden_state.shape,  outputs.text_model_output.pooler_output.shape)
  # break

print(f'\nAccuracy: {correct / total}')
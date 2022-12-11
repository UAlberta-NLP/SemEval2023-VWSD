'''
  Here, we use a Visual Question Answering system to test whether a picture depicts a concept that we care about.
'''

import argparse
import os
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from utils import cos_sim
import termcolor
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='dandelin/vilt-b32-finetuned-vqa')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# In the vanilla case with only one question (Does this depict a {context}?), we score 0.375
# In the ensemble case (with 6 templates) using full context and non-focus context, we score 0.375
# In the ensemble case (with 6 templates) and normalizing "yes" and "no", we score 0.4375

def get_ensemble_questions(context, non_focus_context, focus_word):
  return [
    f'Does this depict a {context}?',
    f'Is there {context} here?',
    f'Does this show {context}?',
    f'Does this depict a {non_focus_context}?',
    f'Is there {non_focus_context} here?',
    f'Does this show {non_focus_context}?',
  ]

def get_question(context, non_focus_context, focus_word):
  return [f'Does this depict a {context}?']

processor = ViltProcessor.from_pretrained(args.model)
model = ViltForQuestionAnswering.from_pretrained(args.model).to(device)

def line_to_tuple(line):
  splits = line.strip().split('\t')
  return tuple(splits)

image_data = [line_to_tuple(l) for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
assert len(image_data) == len(gold_data)

def ask(question: str, image_paths: list):
  images = [Image.open(i).convert('RGB') for i in image_paths]
  questions = [question] * len(images)
  encoding = processor(images, questions, return_tensors="pt").to(device)
  outputs = model(**encoding)
  logits = outputs.logits
  return logits.softmax(dim=0)

yes_id = model.config.label2id['yes']
no_id = model.config.label2id['no']
total = 0
correct = 0
for dat, gold in zip(image_data, gold_data):
  word = dat[0]
  context = dat[1]
  gold = os.path.join(args.image_dir, gold)
  image_paths = [os.path.join(args.image_dir, i) for i in dat[2:]]
  gold_idx = image_paths.index(gold)
  image_paths = sorted(image_paths, key=lambda x: (x.split('.')[1]))
  non_focus = context.replace(word, '').strip()
  questions = get_question(context, non_focus, word)
  print(questions)
  predss = torch.zeros((len(image_paths), model.config.num_labels), device=device)
  for question in questions:
    preds = ask(question, image_paths)
    predss = torch.add(predss, preds)
  # print('\n'.join(preds))
  # print('\n'.join(image_paths))
  # exit()
  yes_preds = predss[:, yes_id]
  no_preds = predss[:, no_id]
  best = image_paths[(yes_preds - no_preds).argmax(dim=0)]
  color = termcolor.colored('right', 'green') if best == gold else termcolor.colored('wrong', 'red')
  print(word, f'"{context}"', best, f'{gold}', '->', color, end='\n\n')
  if gold == best:
    correct += 1
  total += 1

print(f'\nAccuracy: {correct / total}')
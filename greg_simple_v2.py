'''
Algorithm:
  For each instance in the SemEval task with focus word w and context c
    For each image v in the instance
      s(v, c) = similarity between image i and context c
      For each gloss g_i of the focus word w
        s(v, g_i) = similarity between image v and gloss g_i
        s(c, g_i) = similarity between context c and gloss g_i
    Rank the images by the highest total similarity

Formula for the total similarity of an image v:
  w_c * s(v, c) + w_g * MAX_i(w_cg * s(c, g_i) + w_vg * s(v, g_i)) where w_* are tunable parameters 
'''

# Sample command:
# python greg.py -d semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ --model openai/clip-vit-base-patch32

# 500 command:
# python greg.py -d /home/ogezi/ideas/v-wsd/semeval-2023-task-1-V-WSD-train-v1/sample/data.500.txt -g /home/ogezi/ideas/v-wsd/semeval-2023-task-1-V-WSD-train-v1/sample/gold.500.txt -i semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/ --model openai/clip-vit-base-patch32

# Trial command:
# python greg.py -d semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/ --model openai/clip-vit-base-patch32

import argparse
import glob
import os
from time import time
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, BertModel, BertTokenizer
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTFeatureExtractor,
    BertTokenizer,
)
import termcolor
import torch
from tqdm import tqdm
from PIL import ImageFile, Image
from nltk.corpus import wordnet as wn
import numpy as np
import json
import math
from sentence_transformers import SentenceTransformer

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
INST_SIZ = 10

import sys
sys.path.append('.')
from utils import cos_sim, dot_prod_sim, cos_sim_softmax

name = sys.argv[0].replace('.py', '')

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/sample/data.100.txt')
parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/sample/gold.100.txt')
parser.add_argument('--image-dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
parser.add_argument('--model', '-m', default='clip-ViT-B-32')
parser.add_argument('--bert_model', '-bm', default='bert-base-uncased')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--output_results', '-r', default='prediction.txt')
parser.add_argument('--weight_image_context', '-w_ic', default=1., type=float)
parser.add_argument('--weight_context_gloss', '-w_cg', default=1., type=float)
parser.add_argument('--weight_image_gloss', '-w_ig', default=1., type=float)
parser.add_argument('--weight_pool', '-w', default=1., type=float)
parser.add_argument('--pool_func', '-pf', default='max', choices=['max', 'mean'])
parser.add_argument('--wsd_type', '-t', default='consec', choices=['consec', 'amuse'])
parser.add_argument('--wsd_input', '-wi', default='semeval-2023-task-1-V-WSD-train-v1/sample/predictions.100.prob.jsonl')
parser.add_argument('--use_wsd', default=False, action='store_true')
parser.add_argument('--nouns_only', '-n', action='store_true', default=False)
parser.add_argument('--sim', '-s', default='dot_prod_sim', choices=['dot_prod_sim', 'cos_sim', 'cos_sim_softmax'])
parser.add_argument('--temp', default=1., type=float)
args = parser.parse_args()

weight_image_context = args.weight_image_context
weight_pool = args.weight_pool
weight_context_gloss = args.weight_context_gloss
weight_image_gloss = args.weight_image_gloss
pool_func = np.max if args.pool_func == 'max' else np.mean

assert args.wsd_type == 'consec'

if args.output is None:
  default_hyp = weight_image_context == 1. and weight_pool == 1. and weight_context_gloss == 1. and weight_image_gloss == 1. and args.pool_func == 'max'
  if default_hyp:
    hyp_info = '_'
  else:
    hyp_info = f'_w_ic={weight_image_context}_w_cg={weight_context_gloss}_w_ig={weight_image_gloss}_pf={args.pool_func}'
  args.output = f"_logs/{name}{hyp_info}_{int(time())}_{args.model.replace('/', '_')}_log.out"

pos = 'n' if args.nouns_only else None
results = open(args.output_results, 'w')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_wsd:
  parsed_lines = [json.loads(l) for l in open(args.wsd_input).readlines()]
  wsd_in = {int(l['id']): {wn.lemma_from_key(k).synset(): p for k, p in sorted(l['probs'].items(), key=lambda x: x[1], reverse=True)} for l in parsed_lines}

# model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=not True).to(device)
# processor = CLIPProcessor.from_pretrained(args.model)
# tokenizer = CLIPTokenizer.from_pretrained(args.model)
model = SentenceTransformer(args.model)
m_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
bert_model = BertModel.from_pretrained(args.bert_model, low_cpu_mem_usage=True).to(device)
bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)

def get_synsets(word) -> list:
  syns = wn.synsets(word, pos)
  return syns

def sublist_in_list(sub, ls) -> tuple:
  start, end = 0, 0
  sub_sz = len(sub)
  ls_sz = len(ls)
  for idx in range(ls_sz):
    start = idx
    end = idx + sub_sz
    if ls[start:end] == sub:
      return start, end
  return None

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
# assert len(data) == len(gold_data)
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))
correct, total = 0, 0

# TODO: normalize so that sense with many lemmas do not have undue advantage
def lex_sub(focus, context) -> tuple:
  senses = wn.synsets(focus)
  if senses == []:
    return 1, [context]
  gen_contexts = [context]
  lemma_counts = [len(s.lemma_names()) for s in senses]
  hyp_lemma_counts = [[len(h.lemma_names()) for h in (s.hypernyms() + s.instance_hypernyms())] for s in senses]
  max_lemma = max(lemma_counts)
  max_hyp_lemma = 0 # max([max(x) if x != [] else 0 for x in hyp_lemma_counts])
  max_count = max(max_lemma, max_hyp_lemma)
  for idx, sense in enumerate(senses):
    modded_contexts = [context.replace(focus, lemma.replace('_', ' ')) for lemma in sense.lemma_names()] * math.ceil(max_count / lemma_counts[idx])
    modded_contexts = modded_contexts[:max_count]
    assert len(modded_contexts) == max_count
    gen_contexts.extend(modded_contexts)
  # for idx, sense in enumerate(senses):
  #   for jdx, hypernym in enumerate(sense.hypernyms() + sense.instance_hypernyms()):
  #     modded = [context.replace(focus, lemma.replace('_', ' ')) for lemma in hypernym.lemma_names()] * math.ceil(max_count / hyp_lemma_counts[idx][jdx])
  #     modded = modded[:max_count]
  #     assert len(modded) == max_count
  #     gen_contexts.extend(modded)
  # print(len(gen_contexts), gen_contexts)
  return len(gen_contexts), gen_contexts

a, b, c = [], [], []
out = open(args.output, 'w')
sense_counts = []
ranks = []
sim = locals()[args.sim]
iter = tqdm(range(0, len(data)), 'Processing images and text...')
with torch.no_grad():
  for i in iter:
    if i >= len(data):
      break

    instance = data[i]
    gold = gold_data[i]

    def get_def(word, gloss):
      article = 'An' if word.lower()[0] in 'aeiou' else 'A'
      if gloss.lower().startswith('any '):
        pass
      elif gloss.lower().startswith('a '):
        pass
      else:
        gloss = ('an ' if gloss.lower()[0] in 'aeiou' else 'a ') + gloss
      return f'{article} {word} is {gloss}'
      # return gloss

    word, context, *image_paths = instance
    word_tokens = bert_tokenizer(word).input_ids[1:-1]
    synsets = list(set(get_synsets(word)))
    glosses = [get_def(s.lemma_names()[0].replace('_', ' '), s.definition()) for s in synsets]

    images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
    # len_c, extra_contexts = lex_sub(word, context)
    # print(extra_contexts)
    len_c, extra_contexts = 1, [context]
    # extra_contexts = list(set(extra_contexts))
    len_c = len(extra_contexts)
    # bert_inputs = bert_tokenizer(context + glosses, return_tensors='pt', padding=True, truncation=True).to(device)
    # bert_outputs = bert_model(**bert_inputs)
    # hidden_states, last_hidden_states = bert_outputs
    
    #
    # inputs = processor(text=extra_contexts + glosses, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    # outputs = model(**inputs)
    # img_embeds = outputs.image_embeds[:]
    # img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
    #
    img_embeds = model.encode(images)

    len_g = len(glosses)
    # TODO: Check for bugs
    if args.use_wsd and i in wsd_in:
      ss = [s for s, p in wsd_in[i].items()]
      if len(wsd_in[i]) < len_g:
        for s in synsets:
          if s not in ss:
            wsd_in[i][s] = 0.
      assert len(wsd_in[i]) == len_g, f'{wsd_in[i]} {glosses}'
    
    sense_counts.append(len_g)
    #
    # context_embeds = outputs.text_embeds[:len_c]
    # context_embeds /= context_embeds.norm(p=2, dim=-1, keepdim=True)
    # context_embeds = context_embeds.mean(dim=0)
    # context_embeds /= context_embeds.norm()
    # print(context_embeds.shape)
    # gloss_embeds = outputs.text_embeds[len_c:]
    # gloss_embeds = gloss_embeds / gloss_embeds.norm(p=2, dim=-1, keepdim=True)
    #

    context_embeds = m_model.encode([context])
    gloss_embeds = m_model.encode(glosses) if glosses != [] else []

    # bert_inputs.input_ids[0:1][0].tolist()
    # start, end = sublist_in_list(words_tokens[0], bert_inputs.input_ids[:1][0].tolist())
    # mean_focus_word_rep = hidden_states[:l][:, start:end].mean(dim=1)
    # context_bert_embeds = last_hidden_states[:len_c]
    # gloss_bert_embeds = last_hidden_states[len_c:]
    
    # _context_embeds = outputs.text_embeds[:len_c]
    # _context_embeds = context_embeds.mean(dim=0)
    # _img_embeds = outputs.image_embeds[:]
    # _gloss_embeds = outputs.text_embeds[len_c:]

    # _context_embeds = model.get_text_features(inputs.input_ids)[:len_c]
    # _context_embeds = _context_embeds.mean(dim=0)
    # _img_embeds = model.get_image_features(inputs.pixel_values)
    # _gloss_embeds = model.get_text_features(inputs.input_ids)[len_c:]
    # t = (_img_embeds @ _context_embeds.T) @ (_img_embeds @ _gloss_embeds.T)
    # a.append(t / t.norm())
    # t = (_context_embeds @ _gloss_embeds.T)
    # b.append(t / t.norm())
    # c.append(sim(a[i], b[i].T) if (len(a[i])+len(b[i])) > 0 else 1.)

    # print(img_embeds.shape, context_embeds.shape, gloss_embeds.shape)
    sim_image_context = sim(img_embeds, context_embeds.T).T
    sim_context_gloss = sim(context_embeds, gloss_embeds.T).T if glosses != [] else 0.
    # sim_context_gloss_bert = sim(mean_focus_word_rep, gloss_bert_embeds.T).T
    sim_image_gloss = sim(img_embeds, gloss_embeds.T).T if glosses != [] else 0.

    def renorm(probs: dict, temp=args.temp):
      vals = torch.tensor(list(probs.values()))
      logits = torch.log(vals)
      logits /= temp
      return {k: x for k, x in zip(probs.keys(), logits.softmax(dim=0))}

    # pool_func = np.max
    scores = []
    # print(word, len_g)
    # print(glosses)
    # print('sim_image_gloss =', sim_image_gloss)
    # print('sim_context_gloss =', sim_context_gloss)
    # print('sim_image_context =', sim_image_context)
    for idx in range(len(images)):
      if len_g > 0:
        if args.use_wsd and i in wsd_in:
          # print('X')
          # print(sim_image_context.shape, sim_context_gloss.shape, sim_image_gloss.shape)
          probs = wsd_in[i]
          # if idx == 0:
          #   print(word in (list(probs.keys())[0].lemma_names()[0]), list(probs.keys())[0].lemma_names()[0], word, probs)
          #   print(probs)
          probs = renorm(probs)
          # if idx == 0:
          #   print(probs)
          # print(probs, word)
          # print(idx, weight_image_context * sim_image_context[idx].item())
          # print([weight_context_gloss * probs[synsets[g]] for g in range(len_g)])
          # print([weight_image_gloss * sim_image_gloss[idx, g].item() for g in range(len_g)])
          score = weight_image_context * sim_image_context[idx].item() \
            + weight_pool * pool_func([weight_context_gloss * probs[synsets[g]] + weight_image_gloss * sim_image_gloss[idx, g].item() for g in range(len_g)])
        else:
          # print('Y')
          # print(sim_image_context.shape, sim_context_gloss.shape, sim_image_gloss.shape)
          score = weight_image_context * sim_image_context[idx].item() \
            + weight_pool * pool_func([weight_context_gloss * sim_context_gloss[:, g].item() \
            + weight_image_gloss * sim_image_gloss[idx, g].item() for g in range(len_g)])
      else:
      # if True:
        score = weight_image_context * sim_image_context[idx].item()
      scores.append(score)
    scores = torch.tensor(scores)
    # print(scores.argmax(), scores.argsort(descending=True), scores)

    best = image_paths[scores.argmax().item()]
    preds = np.array(image_paths)[scores.argsort(descending=True)].tolist()
    results.write('\t'.join(preds) + '\n')
    results.flush()
    ranks.append(preds.index(gold) + 1)
    total += 1
    is_correct = int(best == gold)
    correct += 1 if is_correct else 0
    color = termcolor.colored('right', 'green') if is_correct else termcolor.colored('wrong', 'red')
    out.write(f'{word} {best} {gold} {image_paths} -> {"right" if is_correct else "wrong"}\n')
    if i % 1 == 0:
      iter.set_postfix({'Accuracy': f'{correct / total:.3f}', 'MRR': f'{np.mean(1 / np.array(ranks)):.3f}'})
    out.flush()

out.write(f'Sense counts: {sense_counts}')
msg = f'\nAccuracy: {correct / total}\nMRR: {np.mean(1 / np.array(ranks))}'
out.write(msg)
print(msg)
out.close()
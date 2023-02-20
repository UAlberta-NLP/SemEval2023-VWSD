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

# Test command:
# python greg.py -d semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.data.v1.txt -g semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial.gold.v1.txt -i semeval-2023-task-1-V-WSD-train-v1/trial_v1/trial_images_v1/ --model openai/clip-vit-base-patch32

# IT: CUDA_VISIBLE_DEVICES=0 python -i greg_v1.py -d /local/storage/ogezi/v-wsd/test/it-en.test.data.txt -g '' -i /local/storage/ogezi/v-wsd/test/test_images/ --model openai/clip-vit-large-patch14 --surround '"' --model /home/ogezi/ideas/v-wsd/latest_1674988116_epoch=1 --bn_glosses /local/storage/ogezi/v-wsd/test/simple.it.test.bn.data.txt -r v1.translations/prediction/prediction.it.txt
# FA: CUDA_VISIBLE_DEVICES=0 python -i greg_v1.py -d /local/storage/ogezi/v-wsd/test/fa-en.test.data.txt -g '' -i /local/storage/ogezi/v-wsd/test/test_images/ --model openai/clip-vit-large-patch14 --surround '"' --model /home/ogezi/ideas/v-wsd/latest_1674988116_epoch=1 --bn_glosses /local/storage/ogezi/v-wsd/test/simple.fa.test.bn.data.txt -r v1.translations/prediction/prediction.fa.txt

import argparse
import glob
import os
from time import time
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, BertModel, BertTokenizer
import termcolor
import torch
from tqdm import tqdm
from PIL import ImageFile, Image
from nltk.corpus import wordnet as wn
import numpy as np
import json
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
INST_SIZ = 10

import sys
sys.path.append('.')
from utils import cos_sim, dot_prod_sim, cos_sim_softmax

name = sys.argv[0].replace('.py', '')

parser = argparse.ArgumentParser()
# parser.add_argument('--data', '-d', default='data/trial.data.txt')
# parser.add_argument('--gold', '-g', default='data/trial.gold.txt')
parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
# parser.add_argument('--image-dir', '-i', default='data/all_images')
parser.add_argument('--image-dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
parser.add_argument('--bert_model', '-bm', default='bert-base-uncased')
parser.add_argument('--instance_batch_size', '-ibs', default=1, type=int, help='This does not follow the conventional meaning of batch size. Kindly take note.')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--output_results', '-r', default='prediction.txt')
parser.add_argument('--weight_image_context', '-w_ic', default=1., type=float)
parser.add_argument('--weight_context_gloss', '-w_cg', default=1., type=float)
parser.add_argument('--weight_image_gloss', '-w_ig', default=1., type=float)
parser.add_argument('--weight_pool', '-w', default=1., type=float)
parser.add_argument('--pool_func', '-pf', default='max', choices=['max', 'mean'])
parser.add_argument('--wsd_type', '-t', default='consec', choices=['consec', 'amuse'])
parser.add_argument('--wsd_input', '-wi', default='consec_train_output/only_nouns/predictions.prob.jsonl')
parser.add_argument('--use_wsd', default=False, action='store_true')
parser.add_argument('--nouns_only', '-n', action='store_true', default=False)
parser.add_argument('--sim', '-s', default='dot_prod_sim', choices=['dot_prod_sim', 'cos_sim', 'cos_sim_softmax'])
parser.add_argument('--surround', default='"')
parser.add_argument('--bn_glosses', required=True)
parser.add_argument('--lang', default=None)
args = parser.parse_args()

weight_image_context = args.weight_image_context
weight_pool = args.weight_pool
weight_context_gloss = args.weight_context_gloss
weight_image_gloss = args.weight_image_gloss
pool_func = np.max if args.pool_func == 'max' else np.mean
if args.bn_glosses:
  bn_glosses = [eval(l.split('\t')[-1]) for l in open(args.bn_glosses).readlines()]

if args.lang is None:
  if 'it' in args.data:
    args.lang = 'it'
  elif 'fa' in args.data:
    args.lang = 'fa'
  else:
    args.lang = 'en'
print(f'Using language as {args.lang}')

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

model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)
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

data = [l.strip().split('\t') for l in open(args.data).readlines()]
if args.gold:
  gold_data = [l.strip() for l in open(args.gold).readlines()]
else:
  gold_data = None
# assert len(data) == len(gold_data)
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))
correct, total = 0, 0

out = open(args.output, 'w')
ibs = args.instance_batch_size
i = 0
j = i + ibs
sense_counts = []
ranks = []
sim = locals()[args.sim]
assert ibs == 1
iter = tqdm(range(0, len(data), ibs), 'Processing images and text...')
with torch.no_grad():
  for i in iter:
    if i >= len(data):
      break

    instance = data[i:j]
    gold = gold_data[i:j] if gold_data is not None else None

    words, contexts, image_pathss, glossess, synsetss, glossess_flat, words_tokens = [], [], [], [], [], [], []
    for inst in instance:
      word, context, *image_paths = inst
      word_tokens = bert_tokenizer(word).input_ids[1:-1]
      wn_synsets = list(set(get_synsets(word)))

      # TODO: Some bn glosses already have the lemma
      def get_def(word, gloss, exs):
        g_lower = gloss.lower()
        article = 'An' if word.lower()[0] in 'aeiou' else 'A'
        if g_lower.startswith('any '):
          pass
        elif g_lower.startswith('a '):
          pass
        else:
          gloss = ('an ' if g_lower[0] in 'aeiou' else 'a ') + gloss
        e = '' if not exs else f'; {exs[0]}'
        # return gloss if g_lower.startswith('a ') else f'{article} {word} is {gloss}{e}'
        return f'{article} {word} is {gloss}{e}'

      # glosses = [f'{s.definition()}' for s in synsets]
      # glosses = [f'{word}: {s.definition()}' for s in synsets]
      # glosses = [get_def(s.lemma_names()[0].replace('_', ' '), s.definition(), s.examples()) for s in wn_synsets]
      glosses = [get_def(word, g, []) for g in bn_glosses[i]]
      words.append(word)
      words_tokens.append(word_tokens)
      if args.lang == 'en':
        # contexts.append(context.replace(word, f"{args.surround}{word}{args.surround}"))
        contexts.append("A " + context.replace(word, f"{args.surround}{word}{args.surround}", 1))
        # contexts.append(f"A photo depicting the {args.surround}{word}{args.surround} in {context}")
      else:
        contexts.append("A " + context)

      glossess.append(glosses)
      synsetss.append(wn_synsets)
      glossess_flat.extend(glosses)
      image_pathss.extend(image_paths)

    # print(words, contexts, image_pathss)
    # len_c, extra_contexts = lex_sub(word, contexts[0])
    # print(extra_contexts, context, contexts)

    len_c = 1
    images = [Image.open(os.path.join(args.image_dir, i)) for i in image_pathss]
    inputs = processor(text=contexts + glossess_flat, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    bert_inputs = bert_tokenizer(contexts + glossess_flat, return_tensors='pt', padding=True, truncation=True).to(device)
    bert_outputs = bert_model(**bert_inputs)
    last_hidden_states = bert_outputs[1]
    
    for k in range(len(instance)):
      img_embeds = outputs.image_embeds[k*INST_SIZ:(k+1)*INST_SIZ]
      img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
      len_g = len(glossess[k])
      wn_synsets = synsetss[k]

      # TODO: Check for bugs
      if args.use_wsd and i in wsd_in:
        ss = [s for s, p in wsd_in[i].items()]
        if len(wsd_in[i]) < len(glossess[k]):
          for s in synsetss[k]:
            if s not in ss:
              wsd_in[i][s] = 0.
        assert len(wsd_in[i]) == len_g, f'{wsd_in[i]} {glossess[k]}'
      
      sense_counts.append(len_g)
      context_embeds = outputs.text_embeds[:len_c]
      context_embeds /= context_embeds.norm(p=2, dim=-1, keepdim=True)
      context_embeds = context_embeds.mean(dim=0)
      context_embeds /= context_embeds.norm()
      gloss_embeds = outputs.text_embeds[len_c:]
      gloss_embeds = gloss_embeds / gloss_embeds.norm(p=2, dim=-1, keepdim=True)
      bert_inputs.input_ids[0:1][0].tolist()

      context_bert_embeds = last_hidden_states[k:k+1]
      gloss_bert_embeds = last_hidden_states[k+1:k+1+len_g]

      sim_image_context = sim(img_embeds, context_embeds.T).T
      sim_context_gloss = sim(context_embeds, gloss_embeds.T).T.unsqueeze(dim=0)

      # NB: This may not work for other languages
      if args.lang == 'en':
        start, end = sublist_in_list(words_tokens[0], bert_inputs.input_ids[k:k+1][0].tolist())
        mean_focus_word_rep = bert_outputs[0][k:k+1][:, start:end].mean(dim=1)
        sim_context_gloss_bert = sim(mean_focus_word_rep, gloss_bert_embeds.T).T
      else:
        sim_context_gloss_bert = sim(context_bert_embeds, gloss_bert_embeds.T).T
      
      sim_image_gloss = sim(img_embeds, gloss_embeds.T).T

      # print('sim_image_context:', sim_image_context.shape)
      # print('sim_context_gloss:', sim_context_gloss.shape)
      # print('sim_image_gloss:', sim_image_gloss.shape)

      pool_func=np.max
      scores = []
      for idx in range(len(images)):
        if len_g > 0:
          if args.use_wsd and i in wsd_in:
            probs = wsd_in[i]
            score = weight_image_context * sim_image_context[idx].item() \
              + weight_pool * pool_func([weight_context_gloss * probs[wn_synsets[g]] + weight_image_gloss * sim_image_gloss[idx, g].item() for g in range(len_g)])
          else:
            score = weight_image_context * sim_image_context[idx].item() \
              + weight_pool * pool_func([weight_context_gloss * sim_context_gloss_bert[:, g].item() + weight_image_gloss * sim_image_gloss[idx, g].item() for g in range(len_g)])
        else:
          score = weight_image_context * sim_image_context[idx].item()
        scores.append(score)
      scores = torch.tensor(scores)

      word = words[k]
      image_paths = image_pathss[k*INST_SIZ:(k+1)*INST_SIZ]
      best = image_paths[scores.argmax().item()]
      preds = np.array(image_paths)[scores.argsort(descending=True)].tolist()
      results.write('\t'.join(preds) + '\n')
      results.flush()
      total += 1
      if gold_data:
        g_k = gold[k] if gold is not None else None
        ranks.append(preds.index(g_k) + 1)
        is_correct = int(best == g_k)
        correct += 1 if is_correct else 0
        color = termcolor.colored('right', 'green') if is_correct else termcolor.colored('wrong', 'red')
        out.write(f'{word} {best} {g_k} {image_paths} -> {"right" if is_correct else "wrong"}\n')
        if i % 1 == 0:
          iter.set_postfix({'Accuracy': f'{correct / total:.3f}', 'MRR': f'{np.mean(1 / np.array(ranks)):.3f}'})
    
    out.flush()
    i += ibs
    j += ibs

out.write(f'Sense counts: {sense_counts}')
if gold_data:
  msg = f'\nAccuracy: {correct / total}\nMRR: {np.mean(1 / np.array(ranks))}'
  out.write(msg)
  print(msg)

out.close()
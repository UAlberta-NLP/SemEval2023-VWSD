"""
  In this experiment, generate captions from a template and a definition for classes from various image dataset
  CIFAR-10, CIFAR-100, MNIST, Fashion MNIST, and ImageNet
"""

from copy import deepcopy
import jax
import jax.numpy as jnp
jax.local_device_count()

import sys
sys.path.append('/home/ogezi/ideas/concept-to-caption/dalle-mini/src')
sys.path.append('/home/ogezi/ideas/concept-to-caption/latent-diffusion/')
sys.path.append('/home/ogezi/ideas/concept-to-caption/latent-diffusion/experiments')
sys.path.append('/home/ogezi/ideas/v-wsd')

from utils import cos_sim
from dalle_mini import DalleBart, DalleBartProcessor

from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel
from flax.jax_utils import replicate
import random
from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
from time import time
import numpy as np
from PIL import Image
import os
from nltk.corpus import wordnet as wn
import argparse
import json
from functools import partial

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, ImageNet
from numba import cuda

import argparse
import glob
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import termcolor
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='data/trial.data.txt')
parser.add_argument('--gold', default='data/trial.gold.txt')
parser.add_argument('--image-dir', default='data/all_images')
parser.add_argument('--model', default='openai/clip-vit-base-patch32')
parser.add_argument('--n_gens', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clip_model = CLIPModel.from_pretrained(args.model) # .to(device)
clip_processor = CLIPProcessor.from_pretrained(args.model)
clip_tokenizer = CLIPTokenizer.from_pretrained(args.model)

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold = [l.strip() for l in open(args.gold).readlines()]
all_images_paths = glob.glob(os.path.join(args.image_dir, '*'))

n_generations = args.n_gens
prompts = [f'A photo of {l[1]}' for l in data]

ts = int(time())
notes = f'Generating images for {len(prompts)} prompts: {n_generations} images per prompt'

print(notes)
print(prompts)

prompts = prompts[:]

DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
DALLE_COMMIT_ID = None
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"

db_model, db_params = DalleBart.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False)
vqgan, vqgan_params = VQModel.from_pretrained(VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False)

db_params = replicate(db_params)
vqgan_params = replicate(vqgan_params)

# model inference
@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
  return db_model.generate(
    **tokenized_prompt,
    prng_key=key,
    params=params,
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    condition_scale=condition_scale,
  )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)

# create a random key
seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)
db_processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

def get_syn(x):
  offset = int(x[3:-1])
  pos = x[-1]
  return wn.synset_from_pos_and_offset(pos, offset)

tokenized_prompts = db_processor(prompts)
tokenized_prompt = replicate(tokenized_prompts)

gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0

images = []
image_names = []
dir = f'gen_data/v-wsd-gen/{ts}'
os.makedirs(dir)

for i in range(max(n_generations // jax.device_count(), 1)):
    key, subkey = jax.random.split(key)
    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        db_params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for j, decoded_img in enumerate(decoded_images):
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)
        name = '{dir}/{d}-{i}.png'.format(dir=dir, d=prompts[j].replace(' ', '_').lower(), i=i)
        image_names.append(name)
        img.save(name)
        print('Saved image @ {}'.format(name))

assert (len(images) / n_generations) == len(data) == len(gold)

n_images = [[] for i in range(len(data))]
images_cp = deepcopy(images)
for i in range(n_generations):
  for j in range(len(data)):
    n_images[j].append(images_cp[(i + 1) * j])

s_images = [[] for i in range(len(data))]
for i, (gen_imgs, context) in enumerate(zip(n_images, prompts)):
  inputs = clip_processor(text=[context], images=gen_imgs, return_tensors="pt", padding=True) # .to(device)
  outputs = clip_model(**inputs)
  logits_per_image = outputs.logits_per_image
  probs = logits_per_image.softmax(dim=0)
  best = gen_imgs[probs.argmax()]
  s_images[i].append(best)

correct, total = 0, 0
for instance, gold_inst, gen_img in zip(data, gold, s_images):
  word, context, *image_paths = instance
  gold_images = [Image.open(os.path.join(args.image_dir, i)) for i in image_paths]
  inputs = clip_processor(text=[f'A photo of {context}'], images=gold_images + gen_img, return_tensors="pt", padding=True) # .to(device)
  outputs = clip_model(**inputs)
  img_e = outputs.image_embeds[:len(gold_images), :]
  gen_img_e = outputs.image_embeds[len(gold_images):, :].mean(dim=0).unsqueeze(dim=0)
  sim = cos_sim(img_e, gen_img_e.T)
  best = image_paths[sim.argmax()]
  total += 1
  correct += 1 if best == gold_inst else 0
  color = termcolor.colored('right', 'green') if best == gold_inst else termcolor.colored('wrong', 'red')
  print(word, best, gold_inst, '->', color)

print(f'\n{n_generations} Accuracy: {correct / total}')

'''
  This program finetunes CLIP on the V-WSD dataset

  nohup python finetune_clip.py --bs 128 --epochs 50 --val_split 0.1 > b32_finetune.out --lr 1e-7 &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 32 --epochs 25 --val_split 0.1 -m openai/clip-vit-base-patch16 --lr 5e-8 > b16_finetune.out &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 24 --epochs 25 --val_split 0.1 -m openai/clip-vit-large-patch14 --lr 1e-8 > l14_finetune.out &
'''

from typing import List
import argparse
import glob
import os
from time import sleep, time
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPFeatureExtractor
import termcolor
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import ImageFile, Image
from nltk.corpus import wordnet as wn
import numpy as np
import json
import math
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
import torchmetrics
from multiprocessing import cpu_count
from pytorch_metric_learning import losses
from utils import cos_sim, dot_prod_sim, cos_sim_softmax, custom_processor, ParallelLoader
import wandb
import einops
import multiprocessing
import ctypes

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
Image.warnings.simplefilter('ignore')
INST_SIZ = 10

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
parser.add_argument('--image_dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--output_results', '-r', default='prediction.txt')
parser.add_argument('--seed', '-s', default=42, type=int)
parser.add_argument('--no_wandb', default=False, action='store_true')
parser.add_argument('--freeze_img_encoder', default=True, action='store_true')
parser.add_argument('--use_smoothing', default=True, action='store_true')
parser.add_argument('--temp', default=12, type=float)

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--val_split', default=0.15, type=float)
parser.add_argument('--grad_acc', default=None, type=int)
args = parser.parse_args()

print('Arguments:')
print(vars(args))

seed_everything(args.seed)
base_name = f"{args.model.replace('/', '_')}_seed={args.seed}_val_split={args.val_split}"
name = f"{base_name}_lr={args.lr}_epochs={args.epochs}_bs={args.bs}_grad_acc={args.grad_acc}_temp={args.temp}"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = [l.strip().split('\t') for l in open(args.data).readlines()]
focus_words, contexts, candidate_data = zip(*[(d[0], d[1], d[2:]) for d in data])
gold_data = [l.strip() for l in open(args.gold).readlines()]

model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

class VWSDDatasetJIT(Dataset):
  def __init__(self, focus_words, contexts, gold_images, candidate_images, loader: ParallelLoader, max_tokens=20):
    self.loader = loader
    self.max_tokens = max_tokens
    self.image_paths = gold_images
    self.contexts = [c.replace(f, f'"{f}"') for f, c in zip(focus_words, contexts)]
    self.candidate_images = candidate_images
    self.gold_labels = [images.index(gold_images[idx]) for idx, images in enumerate(candidate_images)]
    
  def __len__(self):
    return len(self.contexts)
  
  def __getitem__(self, idx) -> tuple:
    joined_names = '_'.join(self.candidate_images[idx])
    pixel_values = self.loader.shared_data[joined_names].to(dtype=torch.float32, device=device) # .squeeze(dim=0)
    context = self.contexts[idx]
    input_ids = processor(text=[context], return_tensors="pt", padding=True, truncation=True).input_ids
    extra_dims = max(0, self.max_tokens - input_ids.size(1))
    input_ids = F.pad(input_ids, (0, extra_dims)).squeeze(dim=0)
    return self.gold_labels[idx], pixel_values, input_ids, idx

# TODO: Use samples with alternative focus words as negatives?
class CLIPWrapper(LightningModule):
  def __init__(self, model, candidate_data, **kwargs):
    super().__init__()
    self.model = model.train()
    self.candidate_data = candidate_data
    self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

  def forward(self, pixel_values, input_ids):
    # outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    if args.freeze_img_encoder:
      with torch.no_grad():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
    text_outputs = model.get_text_features(input_ids=input_ids)
    return image_outputs, text_outputs

  def training_step(self, batch, batch_idx):
    gold_labels, pixel_values, input_ids, _ = batch
    pixel_values = einops.rearrange(pixel_values, 'bs cands c h w -> (bs cands) c h w') # .to(dtype=torch.float32, device=device)
    y_images, y_text = self.forward(pixel_values, input_ids)
    loss = self.compute_loss(y_images, y_text, gold_labels)
    self.log("training_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    return loss

  # def compute_ce_loss(self, y_image, y_text):
  #   co_sim = cos_sim(y_image, y_text.T)
  #   eye = torch.eye(y_image.size(0), device=device)
  #   # loss = torch.norm(co_sim - eye)
  #   loss = F.cross_entropy(co_sim, eye)
  #   return loss

  # def compute_contrastive_loss(self, y_image, y_text, labels):
  #   margin = y_image.size(-1)
  #   dist = (y_image - y_text).norm(dim=-1, p=2)
  #   loss = labels * dist.pow(2) + (1 - labels) * torch.max(margin - dist, 0).pow(2)
  #   return loss

  # contrastive loss function, adapted from
  # https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
  def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
    # print('CL LOGS:', logits.shape)
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

  def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = self.contrastive_loss(similarity)
    image_loss = self.contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
  
  '''
    y_images -> (batch_size x INST_SIZ) x 512
    y_text -> batch_size x 512
  '''
  def compute_loss(self, _y_images, _y_text, gold_labels):
    y_images = _y_images / _y_images.norm(p=2, dim=-1, keepdim=True)
    y_text = _y_text / _y_text.norm(p=2, dim=-1, keepdim=True)
    # print('LOGS:', y_images.shape, y_text.shape)
    logit_scale = self.logit_scale.exp()
    logits_per_text = torch.matmul(y_text, y_images.t()) * logit_scale
    loss = self.clip_loss(logits_per_text)
    # print('LOGS:', logit_scale.shape, logits_per_text.shape, loss.shape)
    return loss

  # def compute_loss(self, _y_images, _y_text, gold_labels):
  #   # similarity -> (batch_size x INST_SIZ) x batch_size
  #   y_images = _y_images / _y_images.norm(p=2, dim=-1, keepdim=True)
  #   y_text = _y_text / _y_text.norm(p=2, dim=-1, keepdim=True)
  #   similarity = dot_prod_sim(_y_images, _y_text.T).T
  #   ideal = torch.zeros_like(similarity)
  #   batch_siz = _y_text.size(0)

  #   # TODO: is there an 'elegant' way to do this?
  #   for idx in range(batch_siz):
  #     label = gold_labels[idx]
  #     if args.use_smoothing:
  #       y_images_inst = _y_images[idx * INST_SIZ:(idx+1) * INST_SIZ]
  #       y_correct = _y_images[(idx * INST_SIZ) + label].unsqueeze(dim=0)
  #       smooth_image_similarity_dist = (dot_prod_sim(y_images_inst, y_correct.T) / args.temp).softmax(dim=1).flatten()
  #       # print(dot_prod_sim(y_images_inst, y_correct.T), smooth_image_similarity_dist, smooth_image_similarity_dist.shape)
  #       # ideal[(idx * INST_SIZ) + label, idx] = 1.
  #       # assert ideal[idx * INST_SIZ:(idx+1) * INST_SIZ, idx].size(0) == image_relative_similarity.size(0)
  #       # assert ideal[idx * INST_SIZ:(idx+1) * INST_SIZ, idx].shape == image_relative_similarity.shape, ideal[idx * INST_SIZ:(idx+1) * INST_SIZ, idx].shape == image_relative_similarity.shape 
  #       ideal[idx * INST_SIZ:(idx+1) * INST_SIZ, idx] = smooth_image_similarity_dist
  #     else:
  #       ideal[(idx * INST_SIZ) + label, idx] = 1.

  #   loss = F.cross_entropy(similarity, ideal)
  #   return loss
  
  def validation_step(self, batch, batch_idx):
    gold_labels, pixel_values, input_ids, *_ = batch
    pixel_values = einops.rearrange(pixel_values, 'bs cands c h w -> (bs cands) c h w') # .to(device)
    _y_images, _y_text = self.forward(pixel_values, input_ids)
    loss = self.compute_loss(_y_images, _y_text, gold_labels)
    self.val_loss = np.append(self.val_loss, loss.mean().cpu())
    self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    y_images = _y_images / _y_images.norm(p=2, dim=-1, keepdim=True)
    y_text = _y_text / _y_text.norm(p=2, dim=-1, keepdim=True)

    batch_siz = input_ids.size(0)
    choices = []
    acc, mrr = 0, 0
    for idx in range(batch_siz):
      sim_context_image = dot_prod_sim(y_text[idx], y_images[idx * INST_SIZ:(idx + 1) * INST_SIZ].T)
      rankings = sim_context_image.argsort(descending=True)
      label = gold_labels[idx]
      choice = rankings[0]
      acc += int(choice == label)
      choices.append(choice)
      ranking = (rankings == label).nonzero() + 1
      mrr += 1 / ranking.item()
    
    choices = torch.tensor(choices)
    acc = acc / batch_siz
    mrr = mrr / batch_siz
    self.val_acc = np.append(self.val_acc, acc)
    self.val_mrr = np.append(self.val_mrr, mrr)

    self.log("val_accuracy", acc, on_epoch=True, on_step=False, prog_bar=True)
    self.log("val_mrr", mrr, on_epoch=True, on_step=False, prog_bar=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
    return optimizer

  def on_validation_epoch_end(self):
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

# model_wrapper = CLIPWrapper(model, candidate_data, val_siz=len(val_loader))
# trainer.validate(model_wrapper, val_loader)

def load_instance(instance_candidates):
  joined_names = '_'.join([p for p in instance_candidates])
  instance_images = [Image.open(os.path.join(args.image_dir, f)) for f in instance_candidates]
  return joined_names, custom_processor(images=instance_images)

loader = ParallelLoader(candidate_data, load_instance)
loader.load() and loader.save()

dataset = VWSDDatasetJIT(focus_words, contexts, gold_data, candidate_data, loader=loader, max_tokens=40)
len_d = len(dataset)
train_split = 1 - args.val_split

splits = [int(train_split * len_d), int(args.val_split * len_d)]
if sum(splits) < len_d:
  splits[0] += 1
elif sum(splits) > len_d:
  splits[0] += 1
# splits = [10, 5]
# splits.append(len(dataset) - sum(splits))

train_set, val_set, *_ = random_split(dataset, splits)
train_sampler = None
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=not True, num_workers=0)

if not args.no_wandb:
  proj_name = 'V-WSD'
  run = wandb.init(project=proj_name)
  run.name = name
  wandb_logger = WandbLogger(project=proj_name)
else:
  wandb_logger = None

model_wrapper = CLIPWrapper(model, candidate_data, val_siz=len(val_loader))
checkpoint_cb = ModelCheckpoint(
  save_top_k=1,
  monitor="val_accuracy", 
  verbose=True,
  filename=args.model.replace('/', '_') + "-{epoch:02d}-{val_accuracy:.4f}"
)

early_stopping_cb = EarlyStopping(
  monitor="val_accuracy", 
  mode="max", 
  patience=args.epochs // 5, 
  verbose=True, 
)

trainer = Trainer(
  logger=wandb_logger, 
  devices=1, 
  accelerator="gpu", 
  max_epochs=args.epochs, 
  check_val_every_n_epoch=1, 
  callbacks=[checkpoint_cb, early_stopping_cb],
  # accumulate_grad_batches=args.grad_acc or int(round(256 / args.bs))
)

trainer.validate(model_wrapper, val_loader)
# trainer.fit(model_wrapper, train_loader, val_loader)
# trainer.validate(model_wrapper, val_loader)

print(f'Best model path: {checkpoint_cb.best_model_path}')
print(f'Best model score: {checkpoint_cb.best_model_score}')
print(f'Best k models: {checkpoint_cb.best_k_models}')
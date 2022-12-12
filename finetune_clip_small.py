# '''
#   This program finetunes CLIP on the V-WSD dataset

#   nohup python finetune_clip.py --bs 128 --epochs 50 --val_split 0.1 > b32_finetune.out --lr 1e-7 &
#   CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 32 --epochs 25 --val_split 0.1 -m openai/clip-vit-base-patch16 --lr 5e-8 > b16_finetune.out &
#   CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 24 --epochs 25 --val_split 0.1 -m openai/clip-vit-large-patch14 --lr 1e-8 > l14_finetune.out &
# '''

# import argparse
# import glob
# import os
# from time import time
# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
# import termcolor
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from PIL import ImageFile, Image
# from nltk.corpus import wordnet as wn
# import numpy as np
# import json
# import math
# from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# from pytorch_lightning import seed_everything, Trainer, LightningModule
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from torch import nn
# import torchmetrics
# from multiprocessing import cpu_count
# from pytorch_metric_learning import losses
# from utils import cos_sim
# import wandb

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = 1000000000
# Image.warnings.simplefilter('ignore')
# INST_SIZ = 10

# import sys
# sys.path.append('.')
# from utils import cos_sim, dot_prod_sim, cos_sim_softmax

# parser = argparse.ArgumentParser()
# parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
# parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
# parser.add_argument('--image_dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
# parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
# parser.add_argument('--output', '-o', default=None)
# parser.add_argument('--output_results', '-r', default='prediction.txt')
# parser.add_argument('--seed', '-s', default=42, type=int)

# parser.add_argument('--lr', default=1e-5, type=float)
# parser.add_argument('--bs', default=32, type=int)
# parser.add_argument('--epochs', default=5, type=int)
# parser.add_argument('--val_split', default=0.15, type=float)
# args = parser.parse_args()

# seed_everything(args.seed)
# base_name = f"{args.model.replace('/', '_')}_seed={args.seed}_val_split={args.val_split}"
# name = f"{base_name}_lr={args.lr}_epochs={args.epochs}_bs={args.bs}"

# IMAGE_MAP = {}

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# data = [l.strip().split('\t') for l in open(args.data).readlines()]
# focus_words, contexts, candidate_data = zip(*[(d[0], d[1], d[2:]) for d in data])
# gold_data = [l.strip() for l in open(args.gold).readlines()]

# model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
# processor = CLIPProcessor.from_pretrained(args.model)
# tokenizer = CLIPTokenizer.from_pretrained(args.model)

# class VWSDDataset(Dataset):
#   def __init__(self, focus_words, contexts, gold_images):
#     images = [Image.open(os.path.join(args.image_dir, f)) for f in gold_images]
#     contexts = [c.replace(f, f'"{f}"') for f, c in zip(focus_words, contexts)]
#     self.inputs = processor(text=contexts, images=images, return_tensors="pt", padding=True, truncation=True)
  
#   def __len__(self):
#     return len(self.contexts)
  
#   def __getitem__(self, idx):
#     return self.inputs[idx].pixel_values, self.inputs[idx].input_ids

# class VWSDDatasetJIT(Dataset):
#   def __init__(self, focus_words, contexts, gold_images, candidate_images, max_tokens=20):
#     self.max_tokens = max_tokens
#     self.image_paths = [os.path.join(args.image_dir, f) for f in gold_images]
#     self.contexts = [c.replace(f, f'"{f}"') for f, c in zip(focus_words, contexts)]
#     self.labels = [images.index(gold_images[idx]) for idx, images in enumerate(candidate_images)]
    
#   def __len__(self):
#     return len(self.contexts)
  
#   def __getitem__(self, idx):
#     image = Image.open(self.image_paths[idx])
#     context = self.contexts[idx]
#     inputs = processor(text=[context], images=[image], return_tensors="pt", padding=True, truncation=True)
#     extra_dims = max(0, self.max_tokens - inputs.input_ids.size(1))
#     padded_tokens = F.pad(inputs.input_ids, (0, extra_dims))
#     return inputs.pixel_values[0], padded_tokens[0], idx, self.labels[idx]

# # TODO: Use samples with alternative focus words as negatives?
# class CLIPWrapper(LightningModule):
#   def __init__(self, model, candidate_paths, **kwargs):
#     global IMAGE_MAP
    
#     super().__init__()
#     self.model = model.train()
#     self.candidate_paths = candidate_paths
#     self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3
    
#     os.makedirs('.cache', exist_ok=True)
#     cache_name = f'cache_{base_name}'
#     self.persist = f'.cache/{cache_name}.pt'
#     self.made_update = False
#     can_load = os.path.exists(self.persist)
#     if can_load:
#       IMAGE_MAP = torch.load(self.persist)
#       print(f'Loading {self.persist}... with {len(IMAGE_MAP)} keys')

#   def forward(self, pixel_values, input_ids):
#     outputs = model(pixel_values=pixel_values, input_ids=input_ids)
#     y_image = outputs.image_embeds
#     y_text = outputs.text_embeds
#     return y_image, y_text

#   def training_step(self, batch, batch_idx):
#     pixel_vals, input_ids, *_ = batch
#     y_image, y_text = self.forward(pixel_vals, input_ids)
#     loss = self.compute_ce_loss(y_image, y_text)
#     self.log("training_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
#     wandb.log({"batch_training_loss": loss})
#     return loss

#   def compute_ce_loss(self, y_image, y_text):
#     co_sim = cos_sim(y_image, y_text.T)
#     eye = torch.eye(y_image.size(0), device=device)
#     # loss = torch.norm(co_sim - eye)
#     loss = F.cross_entropy(co_sim, eye)
#     return loss

#   def compute_contrastive_loss(self, y_image, y_text, labels):
#     margin = y_image.size(-1)
#     dist = (y_image - y_text).norm(dim=-1, p=2)
#     loss = labels * dist.pow(2) + (1 - labels) * torch.max(margin - dist, 0).pow(2)
#     return loss
  
#   def validation_step(self, batch, batch_idx):
#     pixel_vals, input_ids, idxs, labels = batch
#     batch_siz = len(idxs)
#     candidate_paths = self.candidate_paths[idxs.cpu()]
#     y_image, y_text = self.forward(pixel_vals, input_ids)
#     loss = self.compute_ce_loss(y_image, y_text)
#     self.val_loss = np.append(self.val_loss, loss.mean().cpu())
#     self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    
#     all_pixel_vals = []
#     for image_paths in candidate_paths:
#       assert len(image_paths) == INST_SIZ, image_paths
#       names = '_'.join([os.path.basename(p) for p in image_paths])
#       if names not in IMAGE_MAP:
#         self.made_update = True
#         images = []
#         for p in image_paths:
#           images.append(Image.open(p))
#         IMAGE_MAP[names] = processor(images=images, return_tensors='pt').to(device).pixel_values
#       all_pixel_vals.append(IMAGE_MAP[names])
#     all_pixel_vals = torch.cat(all_pixel_vals, dim=0).to(device)

#     candidate_features = model.get_image_features(all_pixel_vals)
      
#     choices = []
#     acc, mrr = 0, 0
#     for i in range(batch_siz):
#       sim_context_image = dot_prod_sim(y_text[i], candidate_features[i*INST_SIZ:(i+1)*INST_SIZ].T)
#       rankings = sim_context_image.argsort(descending=True)
#       label = labels[i]
#       choice = rankings[0]
#       acc += int(choice == label)
#       choices.append(choice)
#       ranking = (rankings == label).nonzero() + 1
#       mrr += 1 / ranking.item()
    
#     choices = torch.tensor(choices)
#     acc = acc / batch_siz
#     mrr = mrr / batch_siz
#     self.val_acc = np.append(self.val_acc, acc)
#     self.val_mrr = np.append(self.val_mrr, mrr)

#     self.log("val_accuracy", acc, on_epoch=True, on_step=False, prog_bar=True)
#     self.log("val_mrr", mrr, on_epoch=True, on_step=False, prog_bar=True)
#     wandb.log({
#       "batch_validation_accuracy": acc, 
#       "batch_validation_loss": loss, 
#       "batch_validation_mrr": mrr,
#     })

#   def configure_optimizers(self):
#     optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
#     return optimizer

#   def on_validation_epoch_end(self):
#     wandb.log({
#       "epoch_validation_accuracy": self.val_acc.mean(), 
#       "epoch_validation_loss": self.val_loss.mean(), 
#       "epoch_validation_mrr": self.val_mrr.mean(),
#     })
#     if self.made_update:
#       torch.save(IMAGE_MAP, self.persist)
#       self.made_update = False
#     self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

# dataset = VWSDDatasetJIT(focus_words, contexts, gold_data, candidate_data, max_tokens=40)
# len_d = len(dataset)
# train_split = 1 - args.val_split

# splits = [int(train_split * len_d), int(args.val_split * len_d)]
# if sum(splits) < len_d:
#   splits[0] += 1
# elif sum(splits) > len_d:
#   splits[0] += 1
# # splits = [66, 66, 12737]

# train_set, val_set, *_ = random_split(dataset, splits)
# train_sampler = None
# train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, sampler=train_sampler, num_workers=0)
# val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=not True, num_workers=0)

# candidate_paths = np.array([[os.path.join(args.image_dir, f) for f in image_group] for image_group in candidate_data])

# # proj_name = 'V-WSD'
# # run = wandb.init(project=proj_name)
# # run.name = name
# # wandb_logger = WandbLogger(project=proj_name)

# model_wrapper = CLIPWrapper(model, candidate_paths=candidate_paths, val_siz=len(val_loader))
# checkpoint_cb = ModelCheckpoint(save_top_k=2, monitor="val_accuracy", verbose=True)
# early_stopping_cb = EarlyStopping(monitor="val_accuracy", mode="max", patience=3, verbose=True)
# trainer = Trainer(
#   # logger=wandb_logger, 
#   devices=1, 
#   accelerator="gpu", 
#   max_epochs=args.epochs, 
#   check_val_every_n_epoch=1, 
#   callbacks=[checkpoint_cb, early_stopping_cb],
#   # accumulate_grad_batches=int(round(128 / args.bs))
# )

# trainer.validate(model_wrapper, val_loader)
# trainer.fit(model_wrapper, train_loader, val_loader)
# trainer.validate(model_wrapper, val_loader)

# print(f'Best model path: {checkpoint_cb.best_model_path}')
# print(f'Best model score: {checkpoint_cb.best_model_score}')
# print(f'Best k models: {checkpoint_cb.best_k_models}')

'''
  This program finetunes CLIP on the V-WSD dataset

  nohup python finetune_clip.py --bs 128 --epochs 50 --val_split 0.1 > b32_finetune.out --lr 1e-7 &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 32 --epochs 25 --val_split 0.1 -m openai/clip-vit-base-patch16 --lr 5e-8 > b16_finetune.out &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_clip.py --bs 24 --epochs 25 --val_split 0.1 -m openai/clip-vit-large-patch14 --lr 1e-8 > l14_finetune.out &
'''

import argparse
import glob
import os
from time import time
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
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
from utils import cos_sim
import wandb

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
Image.warnings.simplefilter('ignore')
INST_SIZ = 10

import sys
sys.path.append('.')
from utils import cos_sim, dot_prod_sim, cos_sim_softmax

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
parser.add_argument('--image_dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
parser.add_argument('--output', '-o', default=None)
parser.add_argument('--output_results', '-r', default='prediction.txt')
parser.add_argument('--seed', '-s', default=42, type=int)

parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--bs', default=32, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--val_split', default=0.15, type=float)
args = parser.parse_args()

seed_everything(args.seed)
base_name = f"{args.model.replace('/', '_')}_seed={args.seed}_val_split={args.val_split}"
name = f"{base_name}_lr={args.lr}_epochs={args.epochs}_bs={args.bs}"

IMAGE_MAP = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = [l.strip().split('\t') for l in open(args.data).readlines()]
focus_words, contexts, candidate_data = zip(*[(d[0], d[1], d[2:]) for d in data])
gold_data = [l.strip() for l in open(args.gold).readlines()]

model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

class VWSDDataset(Dataset):
  def __init__(self, focus_words, contexts, gold_images):
    images = [Image.open(os.path.join(args.image_dir, f)) for f in gold_images]
    contexts = [c.replace(f, f'"{f}"') for f, c in zip(focus_words, contexts)]
    self.inputs = processor(text=contexts, images=images, return_tensors="pt", padding=True, truncation=True)
  
  def __len__(self):
    return len(self.contexts)
  
  def __getitem__(self, idx):
    return self.inputs[idx].pixel_values, self.inputs[idx].input_ids

class VWSDDatasetJIT(Dataset):
  def __init__(self, focus_words, contexts, gold_images, candidate_images, max_tokens=20):
    self.max_tokens = max_tokens
    self.image_paths = [os.path.join(args.image_dir, f) for f in gold_images]
    self.contexts = [c.replace(f, f'"{f}"') for f, c in zip(focus_words, contexts)]
    self.labels = [images.index(gold_images[idx]) for idx, images in enumerate(candidate_images)]
    
  def __len__(self):
    return len(self.contexts)
  
  def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx])
    context = self.contexts[idx]
    inputs = processor(text=[context], images=[image], return_tensors="pt", padding=True, truncation=True)
    extra_dims = max(0, self.max_tokens - inputs.input_ids.size(1))
    padded_tokens = F.pad(inputs.input_ids, (0, extra_dims))
    return inputs.pixel_values[0], padded_tokens[0], idx, self.labels[idx]

# TODO: Use samples with alternative focus words as negatives?
class CLIPWrapper(LightningModule):
  def __init__(self, model, candidate_paths, **kwargs):
    global IMAGE_MAP
    
    super().__init__()
    self.model = model.train() # .to(device)
    self.candidate_paths = candidate_paths
    # TODO: self.loss_func = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3
    
    os.makedirs('.cache', exist_ok=True)
    cache_name = f'cache_{base_name}'
    self.persist = f'.cache/{cache_name}.pt'
    self.made_update = False
    can_load = os.path.exists(self.persist)
    assert IMAGE_MAP == {}
    if can_load:
      IMAGE_MAP = torch.load(self.persist)
      print(f'Loading {self.persist}... with {len(IMAGE_MAP)} keys')

  def forward(self, pixel_values, input_ids):
    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    y_image = outputs.image_embeds
    y_text = outputs.text_embeds
    return y_image, y_text

  def training_step(self, batch, batch_idx):
    pixel_vals, input_ids, *_ = batch
    y_image, y_text = self.forward(pixel_vals, input_ids)
    loss = self.compute_ce_loss(y_image, y_text)
    self.log("training_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    wandb.log({"batch_training_loss": loss})
    return loss

  def compute_ce_loss(self, y_image, y_text):
    co_sim = cos_sim(y_image, y_text.T)
    eye = torch.eye(y_image.size(0), device=device)
    # loss = torch.norm(co_sim - eye)
    loss = F.cross_entropy(co_sim, eye)
    return loss

  def compute_contrastive_loss(self, y_image, y_text, labels):
    margin = y_image.size(-1)
    dist = (y_image - y_text).norm(dim=-1, p=2)
    loss = labels * dist.pow(2) + (1 - labels) * torch.max(margin - dist, 0).pow(2)
    return loss
  
  def validation_step(self, batch, batch_idx):
    pixel_vals, input_ids, idxs, labels = batch
    batch_siz = len(idxs)
    candidate_paths = self.candidate_paths[idxs.cpu()]
    y_image, y_text = self.forward(pixel_vals, input_ids)
    loss = self.compute_ce_loss(y_image, y_text)
    self.val_loss = np.append(self.val_loss, loss.mean().cpu())
    self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    
    candidate_features = []
    for image_paths in candidate_paths:
      assert len(image_paths) == INST_SIZ
      names = '_'.join([os.path.basename(p) for p in image_paths])
      if names not in IMAGE_MAP:
        self.made_update = True
        images = []
        for p in image_paths:
          images.append(Image.open(p))
        inputs = processor(images=images, return_tensors='pt').to(device)
        IMAGE_MAP[names] = model.get_image_features(**inputs)
      candidate_features.append(IMAGE_MAP[names])

    candidate_features = torch.cat(candidate_features, dim=0).to(device)
    print(candidate_features.shape)

    choices = []
    acc, mrr = 0, 0
    for i in range(batch_siz):
      sim_context_image = dot_prod_sim(y_text[i], candidate_features[i*INST_SIZ:(i+1)*INST_SIZ].T)
      rankings = sim_context_image.argsort(descending=True)
      label = labels[i]
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
    wandb.log({
      "batch_validation_accuracy": acc, 
      "batch_validation_loss": loss, 
      "batch_validation_mrr": mrr,
    })

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
    return optimizer

  def on_validation_epoch_end(self):
    wandb.log({
      "epoch_validation_accuracy": self.val_acc.mean(), 
      "epoch_validation_loss": self.val_loss.mean(), 
      "epoch_validation_mrr": self.val_mrr.mean(),
    })
    if self.made_update:
      torch.save(IMAGE_MAP, self.persist)
      self.made_update = False
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

dataset = VWSDDatasetJIT(focus_words, contexts, gold_data, candidate_data, max_tokens=40)
len_d = len(dataset)
train_split = 1 - args.val_split

splits = [int(train_split * len_d), int(args.val_split * len_d)]
if sum(splits) < len_d:
  splits[0] += 1
elif sum(splits) > len_d:
  splits[0] += 1

train_set, val_set = random_split(dataset, splits)
train_sampler = None
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, sampler=train_sampler, num_workers=5)
val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=not True, num_workers=5)

candidate_paths = np.array([[os.path.join(args.image_dir, f) for f in image_group] for image_group in candidate_data])

proj_name = 'V-WSD'
run = wandb.init(project=proj_name)
run.name = name
wandb_logger = WandbLogger(project=proj_name)

model_wrapper = CLIPWrapper(model, candidate_paths=candidate_paths, val_siz=len(val_loader))
checkpoint_cb = ModelCheckpoint(save_top_k=2, monitor="val_accuracy", verbose=True)
early_stopping_cb = EarlyStopping(monitor="val_accuracy", mode="max", patience=3, verbose=True)
trainer = Trainer(
  logger=wandb_logger, 
  devices=1, 
  accelerator="gpu", 
  max_epochs=args.epochs, 
  check_val_every_n_epoch=1, 
  callbacks=[checkpoint_cb, early_stopping_cb],
  accumulate_grad_batches=int(round(128 / args.bs))
)

trainer.validate(model_wrapper, val_loader)
trainer.fit(model_wrapper, train_loader, val_loader)
trainer.validate(model_wrapper, val_loader)

print(f'Best model path: {checkpoint_cb.best_model_path}')
print(f'Best model score: {checkpoint_cb.best_model_score}')
print(f'Best k models: {checkpoint_cb.best_k_models}')
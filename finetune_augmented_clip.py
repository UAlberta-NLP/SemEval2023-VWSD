'''
  This program finetunes CLIP on the V-WSD dataset

  nohup python finetune_augmented_clip.py --bs 128 --epochs 50 --val_split 0.1 > b32_finetune.out --lr 1e-7 &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_augmented_clip.py --bs 32 --epochs 25 --val_split 0.1 -m openai/clip-vit-base-patch16 --lr 5e-8 > b16_finetune.out &
  CUDA_VISIBLE_DEVICES=1 nohup python finetune_augmented_clip.py --bs 24 --epochs 25 --val_split 0.1 -m openai/clip-vit-large-patch14 --lr 1e-8 > l14_finetune.out &
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
from torch.utils.data import Dataset, Subset, DataLoader, random_split, WeightedRandomSampler
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
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000
Image.warnings.simplefilter('ignore')
INST_SIZ = 10

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--e_data', '-e_d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
  parser.add_argument('--e_gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
  parser.add_argument('--e_image_dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1')
  parser.add_argument('--e_val_split', default=0.1, type=float)
  parser.add_argument('--data', '-d', default='greg_augmented_data.txt')
  parser.add_argument('--model', '-m', default='openai/clip-vit-base-patch32')
  parser.add_argument('--output', '-o', default=None)
  parser.add_argument('--output_results', '-r', default='prediction.txt')
  parser.add_argument('--seed', '-s', default=42, type=int)
  parser.add_argument('--no_wandb', default=False, action='store_true')
  parser.add_argument('--freeze_img_encoder', default=True, action='store_true')
  parser.add_argument('--use_smoothing', default=True, action='store_true')
  parser.add_argument('--temp', default=12, type=float)
  parser.add_argument('--surround', default='"')

  parser.add_argument('--lr', default=1e-5, type=float)
  parser.add_argument('--bs', default=32, type=int)
  # parser.add_argument('--e_bs', default=32, type=int)
  parser.add_argument('--epochs', default=5, type=int)
  parser.add_argument('--val_split', default=0.15, type=float)
  parser.add_argument('--grad_acc', default=None, type=int)
  args = parser.parse_args()
  base_name = f"{args.model.replace('/', '_')}_seed={args.seed}_val_split={args.val_split}"
  name = f"{base_name}_lr={args.lr}_epochs={args.epochs}_bs={args.bs}_grad_acc={args.grad_acc}_temp={args.temp}"
  return args, base_name, name

args, base_name, name = get_args()
print('Arguments:')
print(vars(args))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = [l.strip().split('\t') for l in open(args.data).readlines()]
focus_words, contexts, images = zip(*[(d[0], d[1], d[2]) for d in data])

e_data = [l.strip().split('\t') for l in open(args.e_data).readlines()]
e_focus_words, e_contexts, e_candidate_data = zip(*[(d[0], d[1], d[2:]) for d in e_data])
e_gold_data = [l.strip() for l in open(args.e_gold).readlines()]

class VWSDDatasetJIT(Dataset):
  def __init__(self, focus_words, contexts, images, loader: ParallelLoader = None, max_tokens=77):
    self.max_tokens = max_tokens
    self.image_paths = images
    self.contexts = [c.replace(f, f'{args.surround}{f}{args.surround}') for f, c in zip(focus_words, contexts)]
    self.shared_data = {}
    
  def __len__(self):
    return len(self.contexts)
  
  def __getitem__(self, idx) -> tuple:
    image_path = self.image_paths[idx]
    if image_path not in self.shared_data:
      self.shared_data[image_path] = custom_processor(images=[Image.open(image_path)])
    pixel_values = self.shared_data[image_path].to(device=device) # .squeeze(dim=0)
    # pixel_values = custom_processor(images=[Image.open(image_path)]).to(device=device)
    # pixel_values = processor(images=Image.open(image_path), return_tensors='pt') # .to(device=device)
    context = self.contexts[idx]
    input_ids = processor(text=[context], return_tensors="pt", padding=True, truncation=True).input_ids
    extra_dims = max(0, self.max_tokens - input_ids.size(1))
    input_ids = F.pad(input_ids, (0, extra_dims)).squeeze(dim=0)
    return pixel_values, input_ids, idx

# TODO: Use samples with alternative focus words as negatives?
class CLIPWrapper(LightningModule):
  def __init__(self, model, images, **kwargs):
    super().__init__()
    self.model = model.train()
    self.images = images
    self.epoch_starts = {}
    self.logit_scale = torch.nn.Parameter(torch.ones([]) * 2.6592)
    self.val_acc, self.val_loss = (None,) * 2

  def forward(self, pixel_values, input_ids):
    if args.freeze_img_encoder:
      with torch.no_grad():
        image_outputs = model.get_image_features(pixel_values=pixel_values)
    text_outputs = model.get_text_features(input_ids=input_ids)
    return image_outputs, text_outputs

  def training_step(self, batch, batch_idx):
    pixel_values, input_ids, _ = batch
    pixel_values = pixel_values.to(device=device)
    pixel_values = einops.rearrange(pixel_values, 'bs cands c h w -> (bs cands) c h w')
    y_images, y_text = self.forward(pixel_values, input_ids)
    loss = self.compute_loss(y_images, y_text)
    self.log("training_loss", loss, on_epoch=True, on_step=not False, prog_bar=True)
    return loss

  # contrastive loss function, adapted from https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
  def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

  def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = self.contrastive_loss(similarity)
    image_loss = self.contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

  def compute_loss(self, _y_images, _y_text):
    y_images = _y_images / _y_images.norm(p=2, dim=-1, keepdim=True)
    y_text = _y_text / _y_text.norm(p=2, dim=-1, keepdim=True)
    logit_scale = self.logit_scale.exp()
    logits_per_text = torch.matmul(y_text, y_images.t()) * logit_scale
    loss = self.clip_loss(logits_per_text)
    return loss
  
  @torch.no_grad()
  def validation_step(self, batch, batch_idx):
    pixel_values, input_ids, *_ = batch
    pixel_values = pixel_values.to(device=device)
    pixel_values = einops.rearrange(pixel_values, 'bs cands c h w -> (bs cands) c h w')
    _y_images, _y_text = self.forward(pixel_values, input_ids)
    loss = self.compute_loss(_y_images, _y_text)
    self.val_loss = np.append(self.val_loss, loss.mean().cpu())
    self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
    self.log("val_mrr", self.val_mrr, on_epoch=True, prog_bar=True)
    
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=args.lr)

  # def on_validation_epoch_end(self):
  #   self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

  def on_fit_end(self) -> None:
    print(f'\non_fit_end')
    self.do_eval()
  
  def on_epoch_start(self) -> None:
    e = self.current_epoch
    if e not in self.epoch_starts:
      print(f'\non_epoch_start: {e}')
      self.do_eval(do_save=e > 0)
    self.epoch_starts[e] = True

  @torch.no_grad()
  def do_eval(self, do_save=False):
    # e_trainer.validate(e_model_wrapper, e_train_loader)
    dat = e_trainer.validate(e_model_wrapper, e_val_loader)[0]
    self.val_acc = dat['val_acc']
    self.val_mrr = dat['val_mrr']
    if do_save:
      save(model, dir=proj_name, epoch=self.current_epoch, metric=self.val_acc)
    model.to(device)

@torch.no_grad()
def save(model, dir, train=True, epoch=None, metric=None):
  dest = os.path.join(dir, f"{args.model.replace('/', '_').replace(proj_name + '_', '')}_{eye_d}")
  dest += f'_{metric}' if metric is not None else ''
  dest += f'_epoch={epoch}' if epoch is not None else ''
  os.makedirs(dest)
  processor.save_pretrained(dest)
  if not train:
    model.eval()
  model.save_pretrained(dest)
  if not train:
    model.train()
  print(f'Saved to {dest}...')
  return dest

class EvalVWSDDatasetJIT(Dataset):
  def __init__(self, focus_words, contexts, gold_images, candidate_images, loader: ParallelLoader, max_tokens=20):
    self.loader = loader
  # def __init__(self, focus_words, contexts, gold_images, candidate_images, max_tokens=20):
    self.max_tokens = max_tokens
    self.image_paths = gold_images
    self.contexts = [c.replace(f, f'{args.surround}{f}{args.surround}') for f, c in zip(focus_words, contexts)]
    self.candidate_images = candidate_images
    self.gold_labels = [images.index(gold_images[idx]) for idx, images in enumerate(candidate_images)]
    
  def __len__(self):
    return len(self.contexts)
  
  def __getitem__(self, idx) -> tuple:
    joined_names = '_'.join(self.candidate_images[idx])
    pixel_values = self.loader.shared_data[joined_names].to(dtype=torch.float32, device=device)
    # pixel_values = self.shared_data[joined_names].to(dtype=torch.float32, device=device)
    context = self.contexts[idx]
    input_ids = processor(text=[context], return_tensors="pt", padding=True, truncation=True).input_ids
    extra_dims = max(0, self.max_tokens - input_ids.size(1))
    input_ids = F.pad(input_ids, (0, extra_dims)).squeeze(dim=0)
    return self.gold_labels[idx], pixel_values, input_ids, idx

# TODO: Use samples with alternative focus words as negatives?
class EvalCLIPWrapper(LightningModule):
  def __init__(self, model, candidate_data, **kwargs):
    super().__init__()
    self.model = model.train()
    self.candidate_data = candidate_data
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

  def forward(self, pixel_values, input_ids):
    # outputs = model(pixel_values=pixel_values, input_ids=input_ids)
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
  
  '''
    y_images -> (batch_size x INST_SIZ) x 512
    y_text -> batch_size x 512
  '''
  def compute_loss(self, y_images, y_text, gold_labels):
    # similarity -> (batch_size x INST_SIZ) x batch_size
    similarity = dot_prod_sim(y_images, y_text.T).T
    ideal = torch.zeros_like(similarity)
    batch_siz = y_text.size(0)

    # TODO: is there an 'elegant' way to do this?
    for idx in range(batch_siz):
      label = gold_labels[idx]
      ideal[(idx * INST_SIZ) + label, idx] = 1.

    loss = F.cross_entropy(similarity, ideal)
    return loss
  
  @torch.no_grad()
  def validation_step(self, batch, batch_idx):
    gold_labels, pixel_values, input_ids, *_ = batch
    batch_siz = input_ids.size(0)
    pixel_values = einops.rearrange(pixel_values, 'bs cands c h w -> (bs cands) c h w') # .to(device)
    y_images, y_text = self.forward(pixel_values, input_ids)
    y_images = y_images / y_images.norm(p=2, dim=-1, keepdim=True)
    y_text = y_text / y_text.norm(p=2, dim=-1, keepdim=True)
    loss = self.compute_loss(y_images, y_text, gold_labels)
    self.val_loss = np.append(self.val_loss, loss.mean().cpu())
    self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

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

    self.log("val_acc", acc, on_epoch=True, on_step=False, prog_bar=True)
    self.log("val_mrr", mrr, on_epoch=True, on_step=False, prog_bar=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
    return optimizer

  def on_validation_epoch_end(self):
    self.val_acc, self.val_loss, self.val_mrr = [np.array([])] * 3

### E
def e_load_instance(instance_candidates):
  joined_names = '_'.join([p for p in instance_candidates])
  instance_images = [Image.open(os.path.join(args.e_image_dir, f)) for f in instance_candidates]
  return joined_names, custom_processor(images=instance_images)

loader = ParallelLoader(e_candidate_data, e_load_instance)
loader.load() and loader.save()
e_dataset = EvalVWSDDatasetJIT(e_focus_words, e_contexts, e_gold_data, e_candidate_data, loader=loader, max_tokens=77)
# e_dataset = EvalVWSDDatasetJIT(e_focus_words, e_contexts, e_gold_data, e_candidate_data, max_tokens=77)
e_len_d = len(e_dataset)
e_train_split = 1 - args.e_val_split

e_splits = [int(e_train_split * e_len_d), int(args.e_val_split * e_len_d)]
if sum(e_splits) < e_len_d:
  e_splits[0] += 1
elif sum(e_splits) > e_len_d:
  e_splits[0] += 1

generator = torch.Generator()
generator.manual_seed(args.seed)
e_train_set, e_val_set, *_ = random_split(e_dataset, e_splits, generator=generator)
e_train_sampler = None
e_generator = torch.Generator()
e_generator.manual_seed(args.seed)
e_train_loader = DataLoader(e_train_set, batch_size=args.bs, shuffle=True, generator=e_generator, sampler=e_train_sampler, num_workers=0)
e_val_loader = DataLoader(e_val_set, batch_size=args.bs, shuffle=not True, num_workers=0)
e_loader = DataLoader(e_dataset, batch_size=args.bs, shuffle=not True, num_workers=0)
### E

seed_everything(args.seed)

model = CLIPModel.from_pretrained(args.model, low_cpu_mem_usage=True).to(device)
processor = CLIPProcessor.from_pretrained(args.model)
tokenizer = CLIPTokenizer.from_pretrained(args.model)

dataset = VWSDDatasetJIT(focus_words, contexts, images, loader=None, max_tokens=77)

max_workers = 18

def load_instance(image_path):
  return image_path, custom_processor(images=[Image.open(image_path)])

u_images = list(set(images))
kwargs = {
  'total':  len(u_images),
  'desc': "Loading images into memory...",
}
print(f'Loading augmented data with {max_workers} workers...')
dataset.shared_data = dict(process_map(
  load_instance, 
  u_images, 
  max_workers=max_workers, 
  **kwargs
))

# kwargs = {
#   'total':  len(e_candidate_data),
#   'desc': "Loading data into memory...",
# }
# print(f'Loading original data with {max_workers} workers...')
# e_dataset.shared_data = dict(process_map(
#   e_load_instance,
#   e_candidate_data,
#   max_workers=max_workers, 
#   **kwargs
# ))

len_d = len(dataset)
train_split = 1 - args.val_split

splits = [int(train_split * len_d), int(args.val_split * len_d)]
if sum(splits) < len_d:
  splits[0] += 1
elif sum(splits) > len_d:
  splits[0] += 1

def det_split(ds: Dataset, splits: list):
  assert len(ds) == sum(splits)
  ranges = []
  i = 0
  for split in splits:
    split += i
    ranges.append(Subset(ds, range(i, split)))
    i += (split - i)
  return ranges

# train_set, val_set, *_ = random_split(dataset, splits)
train_set, val_set, *_ = det_split(dataset, splits)
train_sampler = None
l_generator = torch.Generator()
l_generator.manual_seed(args.seed)
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, generator=l_generator, num_workers=0)
val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=not True, num_workers=0)

proj_name = 'V-WSD'
eye_d = f"{int(time())}"
if not args.no_wandb:
  run = wandb.init(project=proj_name, id=eye_d)
  run.name = name
  wandb_logger = WandbLogger(project=proj_name)
else:
  wandb_logger = None

model_wrapper = CLIPWrapper(model, images, val_siz=len(val_loader))
e_model_wrapper = EvalCLIPWrapper(model, e_candidate_data, val_siz=len(e_val_loader))

checkpoint_cb = ModelCheckpoint(
  save_top_k=1,
  monitor="val_acc",
  mode="max",
  verbose=True,
  filename=args.model.replace('/', '_') + "-{epoch:02d}-{val_acc:.4f}"
)

early_stopping_cb = EarlyStopping(
  monitor="val_acc", 
  mode="max", 
  patience=3, 
  verbose=True, 
)

e_trainer = Trainer(deterministic=True, logger=wandb_logger, devices=1, accelerator="gpu", max_epochs=0, check_val_every_n_epoch=1)
trainer = Trainer(
  deterministic=True,
  logger=wandb_logger, 
  devices=1, 
  accelerator="gpu", 
  max_epochs=args.epochs, 
  check_val_every_n_epoch=1, 
  callbacks=[checkpoint_cb, early_stopping_cb],
  accumulate_grad_batches=args.grad_acc
)

trainer.fit(model_wrapper, train_loader, val_loader)

# dest = save(trainer, dir=proj_name)
# dest = save(model, dir=proj_name, epoch=None)
# print(f'Saved to {dest}')

print(f'Best model path: {checkpoint_cb.best_model_path}')
print(f'Best model score: {checkpoint_cb.best_model_score}')
print(f'Best k models: {checkpoint_cb.best_k_models}')

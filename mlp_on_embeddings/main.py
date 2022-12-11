# nohup python mlp_on_embeddings/main.py -e 100 -bs 128 > mlp.out &

import argparse
import glob
import os
from time import time
import termcolor
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import torch.nn.functional as F
import torch.multiprocessing as mp

from pytorch_lightning import Trainer, seed_everything, LightningModule

import torchmetrics
from torchsampler import ImbalancedDatasetSampler

import sys
sys.path.append('.')
from utils import cos_sim

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.data.v1.txt')
parser.add_argument('--gold', '-g', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train.gold.v1.txt')
parser.add_argument('--image-dir', '-i', default='semeval-2023-task-1-V-WSD-train-v1/train_v1/train_images_v1/')
parser.add_argument('--batch_size', '-bs', default=32, type=int)
parser.add_argument('--embeddings', '-emb', default='openai_clip-vit-base-patch32.pt')
parser.add_argument('--seed', '-s', default=42, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
args = parser.parse_args()

mp.set_start_method('fork')
seed_everything(args.seed, workers=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = torch.load(args.embeddings)
txt_embs = embeddings['txt']
img_embs = embeddings['img']

data = [l.strip().split('\t') for l in open(args.data).readlines()]
gold_data = [l.strip() for l in open(args.gold).readlines()]
assert len(data) == len(gold_data)
correct, total = 0, 0

bs = args.batch_size
inst_size = 10
dev = 'cuda'

class CustomDataset(Dataset):
  def __init__(self, data=data, gold_data=gold_data, embeddings=embeddings, dev=dev):
    txt_emb_dict = embeddings['txt']
    img_emb_dict = embeddings['img']
    self.txt_embs = []
    self.img_embs = []
    self.dev = dev

    self.labels = []
    for instance, gold in zip(data, gold_data):
      word, context, *image_paths = instance
      for i in range(len(image_paths)):
        k = image_paths[i]
        self.img_embs.append(img_emb_dict[k])
        self.txt_embs.append(txt_emb_dict[context])
        self.labels.append(1 if gold == k else 0)
    assert len(self.txt_embs) == len(self.img_embs)
    assert len(self.labels) == len(self.img_embs)

  def __getitem__(self, index) -> tuple:
    return (
      torch.cat((self.txt_embs[index], self.img_embs[index])).to(self.dev),
      torch.tensor([self.labels[index]], dtype=torch.float),
    )

  def __len__(self) -> int:
    return len(self.txt_embs)

class LogisticRegression(LightningModule):
  def __init__(self, in_features, dev=dev):
    super().__init__()
    self.dev = dev
    self.linear1 = nn.Linear(in_features, 1024)
    self.linear2 = nn.Linear(1024, 1)
    self.acc = torchmetrics.Accuracy(threshold=0.5)
    self.p = torchmetrics.Precision(threshold=0.5)
    self.r = torchmetrics.Recall(threshold=0.5)
    self.confusion = torchmetrics.ConfusionMatrix(num_classes=2, threshold=0.5)
    self.confusions = torch.zeros(2, 2, device=self.dev)

  def forward(self, x):
    y_hat = torch.sigmoid(
      self.linear2(self.linear1(x))
    )
    return y_hat

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.binary_cross_entropy(y_hat, y)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.forward(x)
    loss = F.binary_cross_entropy(y_hat, y)
    y = y.to(torch.int)
    self.log("val_loss", loss)
    self.log("val_accuracy", self.acc(y_hat, y))
    self.log("val_precision", self.p(y_hat, y))
    self.log("val_recall", self.r(y_hat, y))
    self.confusions = torch.add(self.confusions, self.confusion(y_hat, y))

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  def on_validation_epoch_end(self):
    print('Confusion Matrix')
    print(self.confusions)
    self.confusions = torch.zeros(2, 2, device=self.dev)

dataset = CustomDataset()
splits = [int(0.8 * len(dataset)), int(0.2 * len(dataset))]
train_set, val_set = random_split(dataset, splits)
ones = [1 for i in train_set if i[-1] == 1]

freq_dict = {1: len(ones), 0: len(train_set) - len(ones)}
weights = [1 / freq_dict[i[-1].item()] for i in train_set]
train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_set))
# train_sampler = ImbalancedDatasetSampler(train_set, labels=np.array(train_set.dataset.labels)[train_set.indices], indices=train_set.indices, num_samples=len(train_set))

train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler)
val_loader = DataLoader(val_set, batch_size=bs, shuffle=True)

model = LogisticRegression(in_features=1024).to(dev)
trainer = Trainer(devices=1, accelerator="gpu", max_epochs=args.epochs)

trainer.validate(model, val_loader)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)
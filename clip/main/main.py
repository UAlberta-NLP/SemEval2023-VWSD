#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# public
import torch
# private
from config import Config
from src.trainer import CLIPTrainer
from src.utils import helper


class CLIP(object):
    """docstring for CLIP"""
    def __init__(self):
        super(CLIP, self).__init__()
        self.config = Config()
        self.update_config()
        self.initialize()

    def update_config(self):
        # setup device
        self.config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self):
        # setup random seed
        helper.set_seed(self.config.seed)
        # trainer
        self.trainer = CLIPTrainer(self.config)

def main():
    clip = CLIP()
    clip.trainer.train()

if __name__ == '__main__':
      main()
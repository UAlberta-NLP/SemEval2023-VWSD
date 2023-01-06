#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, argparse
# public


# helper function
def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_args():
    parser = argparse.ArgumentParser()
    # random seed
    parser.add_argument('--seed', type=int, default=0)
    # data
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=12)
    # train
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=256)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--load_ckpt', type=str2bool, default=False)
    # validation
    parser.add_argument('--val_patience', type=int, default=32)  # w.r.t. epoch
    # pretrained models
    parser.add_argument('--plm', type=str, default='clip-vit-base-patch32')  # pretrained language model
    parser.add_argument('--pvm', type=str, default='clip-vit-base-patch32')  # pretrained vision model
    # save as argparse space
    return parser.parse_known_args()[0]


class Config():
    # config settings
    def __init__(self):
        super(Config, self).__init__()
        self.update_config(**vars(init_args()))
        
    def update_config(self, **kwargs):
        # load config from parser
        for k,v in kwargs.items():
            setattr(self, k, v)
        # I/O
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        # data
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        self.TRAIN_PATH = os.path.join(self.DATA_PATH, 'semeval-2023-task-1-V-WSD-train-v1', 'train_v1')
        self.TRAIN_IMG_PATH = os.path.join(self.TRAIN_PATH, 'train_images_v1')
        self.TRAIN_TEXT_FILE = os.path.join(self.TRAIN_PATH, 'train.data.v1.txt')
        self.TRAIN_LABEL_FILE = os.path.join(self.TRAIN_PATH, 'train.gold.v1.txt')
        self.SPLIT_PKL = os.path.join(self.TRAIN_PATH, 'split.pkl')
        # pretrained models
        self.MODEL_PATH = os.path.join(self.RESOURCE_PATH, 'models')
        self.PLM_PATH = os.path.join(self.MODEL_PATH, self.plm)
        self.PVM_PATH = os.path.join(self.MODEL_PATH, self.pvm)
        # checkpoints
        self.CKPT_PATH = os.path.join(self.RESOURCE_PATH, 'ckpts', str(self.seed))
        os.makedirs(self.CKPT_PATH, exist_ok=True)
        self.CKPT_PT = f'{self.plm}_{self.pvm}.pt'
        self.CKPT_PT = os.path.join(self.CKPT_PATH, self.CKPT_PT)
        # log
        self.LOG_PATH = os.path.join(self.RESOURCE_PATH, 'log', str(self.seed))
        os.makedirs(self.LOG_PATH, exist_ok=True)
        # to save log in txt
        self.LOG_TXT = f'{self.plm}_{self.pvm}.txt'
        self.LOG_TXT = os.path.join(self.LOG_PATH, self.LOG_TXT)
        os.remove(self.LOG_TXT) if os.path.exists(self.LOG_TXT) else None
        # to save log in pickle
        self.LOG_PKL = f'{self.plm}_{self.pvm}.pkl'
        self.LOG_PKL = os.path.join(self.LOG_PATH, self.LOG_PKL)
        os.remove(self.LOG_PKL) if os.path.exists(self.LOG_PKL) else None
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import random, pickle
# public
import torch
import transformers
import numpy as np


def set_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # transformers
    transformers.set_seed(seed)
    # cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def txt2list(path: str) -> list:
    with open(path, 'r') as f:
        return f.read().splitlines()

def save_pickle(path, obj):
    """
    To save a object as a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    """
    To load object from pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_optim_params(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    parameters = []
    parameters.append(
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': config.weight_decay})
    parameters.append(
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0})
    return parameters
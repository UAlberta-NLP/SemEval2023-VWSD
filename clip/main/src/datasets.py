#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, random
# public
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
# private
from src.utils import helper


class VWSDDataset(Dataset):
    """docstring for Dataset"""
    def __init__(self, mode, config):
        """
            mode (str): train, val, or test
        """
        super(Dataset, self).__init__()
        self.mode = mode
        self.config = config
        _, self.xs, self.ys, _ = self.get_data()
        self.data_size = len(self.xs)
        self.val_size = round(self.data_size * self.config.val_rate)
        # train val split
        self.split_data()

    def __len__(self): 
        return self.data_size

    def get_data(self):
        texts_list = helper.txt2list(self.config.TRAIN_TEXT_FILE)
        labels_list = helper.txt2list(self.config.TRAIN_LABEL_FILE)
        words_list, contexts_list, imgs_list = [], [], []
        for text in texts_list:
            text = text.split('\t')
            words_list.append(text[0])
            contexts_list.append(text[1])
            imgs_list.append(text[2:])
        return words_list, contexts_list, labels_list, imgs_list

    def split_data(self):
        # data size for train and val
        split_dict = {}
        # initialize data splits
        if not os.path.exists(self.config.SPLIT_PKL):
            idx_list = list(range(self.data_size))
            random.shuffle(idx_list)
            split_dict['val'] = set(idx_list[:self.val_size])
            split_dict['train'] = set(idx_list[self.val_size:])
            helper.save_pickle(self.config.SPLIT_PKL, split_dict)
        # load data splits
        else:
            split_dict = helper.load_pickle(self.config.SPLIT_PKL)
        # split data
        idx_set = split_dict[self.mode]
        self.xs, self.ys = map(
            list, zip(*[(x, y) for i, (x, y) in enumerate(zip(self.xs, self.ys)) if i in idx_set]))
        self.data_size = len(self.xs)

    def get_img(self, img_name):
        img_file = os.path.join(self.config.TRAIN_IMG_PATH, img_name)
        # TODO: RuntimeError: Unsupported color conversion request
        try:
            return read_image(img_file, mode=ImageReadMode.RGB)
        except:
            return None
        
    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.get_img(self.ys[idx])
        return x, y
#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import os, sys, copy, datetime, logging
# public
import torch
from torch.utils import data as torch_data
from transformers import (
    CLIPModel
    , AutoTokenizer
    , AutoFeatureExtractor
    , VisionTextDualEncoderProcessor
    , get_linear_schedule_with_warmup
)
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
# private
from src.utils import helper
from src.datasets import VWSDDataset


# helper function
def init_logger(config):
    """initialize the logger"""
    file_handler = logging.FileHandler(filename=config.LOG_TXT)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding='utf-8'
        , format='%(asctime)s | %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S'
        , level=logging.INFO
        , handlers=handlers
        )
    global logger
    logger = logging.getLogger(__name__)
    return logger


class BaseTrainer(object):
    """docstring for BaseTrainer"""
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        self.config = config
        self.__update_config()
        self.__initialize()
        self.setup_log_dict()

    def __update_config(self):
        # global configurations always not changed
        self.config.shuffle = True  # to shuffle the training set
        self.config.pin_memory = True  # pin memory for dataloader
        self.config.drop_last = True  # drop the last training batch

    def __initialize(self):
        # some default variables and placeholders
        self.step, self.epoch, self.val_epoch = 0, 0, 0
        self.done = False
        self.dataloader_dict = {}
        self.valid_epoch = 0

    def setup_log_dict(self):
        self.log_dict = {}
        for mode in ['train', 'val']:
            self.log_dict[mode] = {}
            self.log_dict[mode]['eval'] = []
            self.log_dict[mode]['best_eval'] = []
        self.log_dict['start_time'] = datetime.datetime.now()
        self.log_dict['best_val_metric'] = float('inf')

    def save_ckpt(self):
        # define the checkpoin to be saved
        checkpoint_to_save = {
        'step': self.step
        , 'epoch': self.epoch
        , 'val_epoch': self.val_epoch
        , 'log_dict': self.log_dict
        , 'model': self.model.state_dict()
        , 'optimizer': self.optimizer.state_dict()
        , 'scheduler': self.scheduler.state_dict()
        }
        # save the check point
        torch.save(checkpoint_to_save, self.config.CKPT_PT)

    def load_ckpt(self):
        ckpt_to_load =  torch.load(self.config.CKPT_PT, map_location=self.config.device) 
        self.step = ckpt_to_load['step']
        self.epoch = ckpt_to_load['epoch']
        self.val_epoch  = ckpt_to_load['val_epoch']
        self.log_dict = ckpt_to_load['log_dict']
        self.model.load_state_dict(ckpt_to_load['model'])
        self.optimizer.load_state_dict(ckpt_to_load['optimizer'])
        self.scheduler.load_state_dict(ckpt_to_load['scheduler'])


class CLIPTrainer(BaseTrainer):
    """docstring for CLIPTrainer"""
    def __init__(self, config, **kwargs):
        super(CLIPTrainer, self).__init__(config)
        self.update_config(**kwargs)
        self.initialize()
        self.setup_dataloader()
        # restore the trainer from the checkpint if needed
        if self.config.load_ckpt:
            self.load_ckpt()
            logger.info('Trainer restored from {}.'.format(self.config.CKPT_PT))

    def update_config(self, **kwargs):
        # update configuration accordingly
        for k,v in kwargs.items():
            setattr(self.config, k, v)

    def initialize(self):
        # enable tokenizer multi-processing
        if self.config.num_workers > 0:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.PLM_PATH)
        # feature extractor
        extractor = AutoFeatureExtractor.from_pretrained(self.config.PVM_PATH)
        # processor
        self.processor = VisionTextDualEncoderProcessor(extractor, tokenizer)
        # model graph
        if self.config.plm == self.config.pvm:
            self.model = CLIPModel.from_pretrained(self.config.PLM_PATH).to(self.config.device)
        else:
            raise NotImplementedError
        self.config.num_parameters = f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}'
        # optimizer
        optim_params = helper.get_optim_params(self.model, self.config)
        self.optimizer = torch.optim.AdamW(optim_params, lr=self.config.learning_rate)
        # save config
        self.log_dict['config'] = self.config.__dict__

    def collate_fn(self, data):
        """
        a customized collate function used in the data loader 
        """
        data.sort(key=len, reverse=True)
        xs, ys = map(list, zip(*data))
        xs, ys = map(list, zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]))
        # dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
        inputs = self.processor(
            text=xs
            , images=ys
            , return_tensors='pt'
            , padding=True
            )
        return inputs

    def setup_dataloader(self):
        # train dataset
        dataset = VWSDDataset('train', self.config)
        self.config.train_size = len(dataset)
        dataloader = torch_data.DataLoader(
            dataset
            , batch_size=self.config.batch_size
            , collate_fn=self.collate_fn
            , shuffle=self.config.shuffle
            , num_workers=self.config.num_workers
            , pin_memory=self.config.pin_memory
            , drop_last=self.config.drop_last
            )
        self.dataloader_dict['train'] = dataloader
        # validation dataset
        dataset = VWSDDataset('val', self.config)
        self.config.val_size = len(dataset)
        dataloader = torch_data.DataLoader(
            dataset
            , batch_size=self.config.batch_size
            , collate_fn=self.collate_fn
            , shuffle=False
            , num_workers=self.config.num_workers
            , pin_memory=self.config.pin_memory
            , drop_last=False
            )
        self.dataloader_dict['val'] = dataloader
        # update config
        self.config.max_steps = int(self.config.max_epochs*(self.config.train_size/self.config.batch_size))
        # get scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer
            , num_warmup_steps=self.config.warmup_steps
            , num_training_steps=self.config.max_steps
            )
        # initialize logger
        init_logger(self.config)
        logger.info('Initialized logger.')

    def one_epoch(self, mode):
        # initialization
        self.mode = mode
        # dataloader
        dataloader = tqdm(self.dataloader_dict[mode])
        epoch_loss, epoch_steps = 0., 0
        with logging_redirect_tqdm():
            for inputs in dataloader:
                # move to device
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                # model feedward
                outputs = self.model(**inputs, return_loss=True)
                loss = outputs.loss
                if self.model.training:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.step += 1
                loss = loss.item()
                dataloader.set_description('{} Loss:{:.4f} LR:{:.6f}'.format(
                    mode.capitalize(), loss, self.scheduler.get_last_lr()[0]))
                # post processing
                epoch_loss += loss
                epoch_steps += 1
                # break
        return epoch_loss / epoch_steps

    def train(self):
        # show configurations
        logger.info('*Configurations:*')
        for k, v in self.config.__dict__.items():
            logger.info(f'\t{k}: {v}')
        # go
        logger.info("Start training...")
        while True:
            self.epoch += 1
            self.model.train()
            loss = self.one_epoch('train')
            self.log_dict['train']['eval'] += [[self.step, self.epoch, loss]]
            logger.info(f'Train Epoch:{self.epoch} Step:{self.step} Loss: {loss}')
            # validation
            self.eval('val')
            # check if early stop
            self.early_stopping()
            # maximum training steps:
            if self.val_epoch > self.config.val_patience or self.step > self.config.max_steps:
                # save log
                self.log_dict['end_time'] = datetime.datetime.now()
                helper.save_pickle(self.config.LOG_PKL, self.log_dict)
                logger.info('Log saved as {}.'.format(self.config.LOG_PKL))
                logger.info('Training completed.')
                break

    def eval(self, mode):
        self.model.eval()
        with torch.no_grad():
            loss = self.one_epoch(mode)
            # evaluation
            self.log_dict[mode]['eval'] += [[self.step, self.epoch, loss]]
            # save results
            logger.info('{} Epoch:{} Valid: {}/{} Step:{} Loss: {}'.format(
                mode.capitalize(), self.epoch, self.val_epoch, self.config.val_patience, self.step, loss))

    def early_stopping(self):
        # check if early stop based on the validation
        if self.log_dict['val']['eval'][-1][-1] < self.log_dict['best_val_metric']:
            logger.info('Got the best validation so far! ({} < {})'.format(
                    self.log_dict['val']['eval'][-1][-1], self.log_dict['best_val_metric']))
            # update the validation best record
            self.log_dict['best_val_metric'] = self.log_dict['val']['eval'][-1][-1]
            # best evaluation
            self.log_dict['val']['best_eval'] =  copy.deepcopy(self.log_dict['val']['eval'])
            # reset validation epoch
            self.val_epoch = 0
            # save trainer
            self.save_ckpt()
            logger.info('Trainer saved as {}.'.format(self.config.CKPT_PT))
            # save log
            self.log_dict['end_time'] = datetime.datetime.now()
            helper.save_pickle(self.config.LOG_PKL, self.log_dict)
            logger.info('Log saved as {}.'.format(self.config.LOG_PKL))
        else:
            self.val_epoch += 1
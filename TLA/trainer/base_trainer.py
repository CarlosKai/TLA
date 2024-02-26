# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.utils import accuracy, AverageMeter

from utils.utils import save_model, write_log
from utils.lr_scheduler import inv_lr_scheduler
from datasets import *
from sklearn.metrics import f1_score


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.channels = 0

        logging.info(f'--> trainer: {self.__class__.__name__}')

        self.setup()
        self.build_datasets()
        self.build_models()
        self.resume_from_ckpt()

    def build_models(self):
        pass
    def resume_from_ckpt(self):
        pass

    def setup(self):
        self.start_ite = 0
        self.ite = 0
        self.best_acc = 0.
        self.tb_writer = SummaryWriter(self.cfg.TRAIN.OUTPUT_TB)
        self.best_macro_f1 = 0.

    def build_datasets(self):
        logging.info(f'--> building dataset from: {self.cfg.DATASET.NAME}')
        self.dataset_loaders = {}

        # dataset loaders
        if self.cfg.DATASET.NAME == 'EEG':
            dataset = EEG
        elif self.cfg.DATASET.NAME == 'HHAR':
            dataset = HHAR
        elif self.cfg.DATASET.NAME == 'WISDM':
            dataset = WISDM
        elif self.cfg.DATASET.NAME == 'HAR':
            dataset = WISDM
        else:
            raise ValueError(f'Dataset {self.cfg.DATASET.NAME} not found')

        # SOURCE
        # loading path
        train_src_path = torch.load(os.path.join(self.cfg.DATASET.ROOT, "train_" + self.cfg.DATASET.SOURCE + ".pt"))
        test_src_path = torch.load(os.path.join(self.cfg.DATASET.ROOT, "test_" + self.cfg.DATASET.SOURCE + ".pt"))
        # Loading datasets
        train_src_dataset = dataset(train_src_path, status='train')
        test_src_dataset = dataset(test_src_path, status='val')

        self.channels = train_src_dataset.num_channels

        # Dataloaders
        self.dataset_loaders['source_train'] = DataLoader(
            train_src_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_SOURCE,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['source_test'] = DataLoader(
            test_src_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )


        # TARGET
        # loading path .pt file
        train_tar_path = torch.load(os.path.join(self.cfg.DATASET.ROOT, "train_" + self.cfg.DATASET.TARGET + ".pt"))
        test_tar_path = torch.load(os.path.join(self.cfg.DATASET.ROOT, "test_" + self.cfg.DATASET.TARGET + ".pt"))
        # Loading datasets
        train_tar_dataset = dataset(train_tar_path, status='train')
        test_tar_dataset = dataset(test_tar_path, status='val')
        # Dataloaders
        self.dataset_loaders['target_train'] = DataLoader(
            train_tar_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TARGET,
            shuffle=True,
            num_workers=self.cfg.WORKERS,
            drop_last=True
        )
        self.dataset_loaders['target_test'] = DataLoader(
            test_tar_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=self.cfg.WORKERS,
            drop_last=False
        )

        self.len_src = len(self.dataset_loaders['source_train'])    # 这里的数量是已经经过划分为batch之后的dataloader数量
        self.len_tar = len(self.dataset_loaders['target_train'])
        logging.info(f'    source {self.cfg.DATASET.SOURCE}: {self.len_src}'
                     f'/{len(self.dataset_loaders["source_test"])}')
        logging.info(f'    target {self.cfg.DATASET.TARGET}: {self.len_tar}'
                     f'/{len(self.dataset_loaders["target_test"])}')

    def model_parameters(self):  # 计算模型的大小
        for k, v in self.registed_models.items():
            logging.info(f'    {k} paras: '
                         f'{(sum(p.numel() for p in v.parameters()) / 1e6):.2f}M')

    def build_optim(self, parameter_list: list):
        self.optimizer = optim.SGD(
            parameter_list,
            lr=self.cfg.TRAIN.LR,
            momentum=self.cfg.OPTIM.MOMENTUM,
            weight_decay=self.cfg.OPTIM.WEIGHT_DECAY,
            nesterov=True
        )
        self.lr_scheduler = inv_lr_scheduler



    def train(self):
        # start training
        for _, v in self.registed_models.items():
            v.train()
        for self.ite in range(self.start_ite, self.cfg.TRAIN.TTL_ITE):
            # test
            if self.ite % self.cfg.TRAIN.TEST_FREQ == self.cfg.TRAIN.TEST_FREQ - 1 and self.ite != self.start_ite:
                self.base_net.eval()
                self.test()
                self.base_net.train()

            self.current_lr = self.lr_scheduler(
                self.optimizer,
                # ite_rate=self.ite / self.cfg.TRAIN.TTL_ITE * self.cfg.METHOD.HDA.LR_MULT,
                ite_rate=self.ite / self.cfg.TRAIN.TTL_ITE,
                lr=self.cfg.TRAIN.LR,
            )

            # dataloader
            if self.ite % self.len_src == 0 or self.ite == self.start_ite:  # 如果达到源域数据集的末尾，重新初始化源域数据加载器。
                iter_src = iter(self.dataset_loaders['source_train'])
            if self.ite % self.len_tar == 0 or self.ite == self.start_ite:
                iter_tar = iter(self.dataset_loaders['target_train'])

            # forward one iteration
            data_src = iter_src.__next__()
            data_tar = iter_tar.__next__()
            self.one_step(data_src, data_tar)
            # if self.ite % self.cfg.TRAIN.SAVE_FREQ == 0 and self.ite != 0:  # 保存模型
            #     self.save_model(is_best=False, snap=True)


    def display(self, data: list):
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | lr: {self.current_lr:.5f} '
        # update
        for _str in data:
            log_str += '| {} '.format(_str)
        logging.info(log_str)

    def update_tb(self, data: dict):
        for k, v in data.items():
            self.tb_writer.add_scalar(k, v, self.ite)

    def step(self, loss_ttl):
        self.optimizer.zero_grad()
        loss_ttl.backward()
        self.optimizer.step()

    def test(self):
        logging.info('--> testing on source_test')
        src_acc, src_macro_f1 = self.test_func(self.dataset_loaders['source_test'], self.base_net)
        logging.info('--> testing on target_test')
        tar_acc, tar_macro_f1 = self.test_func(self.dataset_loaders['target_test'], self.base_net)
        is_best = False
        # if tar_acc > self.best_acc: # 希望准确率尽量高
        # if  tar_macro_f1 > self.best_macro_f1:    # 希望f1分数尽量高
        if tar_acc > self.best_acc or tar_macro_f1 > self.best_macro_f1:
            self.best_acc = max(self.best_acc, tar_acc)
            self.best_macro_f1 = max(self.best_macro_f1, tar_macro_f1)
            is_best = True

        # display
        log_str = f'I:  {self.ite}/{self.cfg.TRAIN.TTL_ITE} | src_acc: {src_acc:.3f} | tar_acc: {tar_acc:.3f} | tar_macrof1 : {tar_macro_f1:.3f}' \
                  f'best_acc: {self.best_acc:.3f} | best_macro_f1: {self.best_macro_f1:.3f}'
        logging.info(log_str)

        # save results
        log_dict = {
            'I': self.ite,
            'src_acc': src_acc,
            'tar_acc': tar_acc,
            'tar_macro_f1': tar_macro_f1,
            'best_acc': self.best_acc,
            'best_macro_f1': self.best_macro_f1
        }
        write_log(self.cfg.TRAIN.OUTPUT_RESFILE, log_dict)

        # tensorboard
        self.tb_writer.add_scalar('tar_acc', tar_acc, self.ite)
        self.tb_writer.add_scalar('src_acc', src_acc, self.ite)

        self.save_model(is_best = is_best)




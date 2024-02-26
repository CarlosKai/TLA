# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import torch
import random
from PIL import Image
from PIL import ImageFile
import numpy as np
from torch.utils.data import Dataset
from datasets.reader import read_images_labels
import torch.nn as nn


class WISDM(Dataset):
    def __init__(self, dataset, status):
        super(WISDM, self).__init__()

        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        X_train = dataset["samples"]
        Y_train = dataset["labels"]

        if len(X_train.shape) < 3:  # 需要三维的向量
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):  # 转换为torch
            X_train = torch.from_numpy(X_train)
            Y_train = torch.from_numpy(Y_train).long()  # 将数据类型转换为64位整数

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)  # 用于指定新的维度顺序

        self.x_data = X_train
        self.y_data = Y_train

        self.num_channels = X_train.shape[1]

        self.len = X_train.shape[0]  # 时序样本数
        self.domain_id = [0] * len(self.y_data)

    def __getitem__(self, index):

        label = self.y_data[index].long()
        time_image = self.x_data[index]
        image = time_image.float()
        domain = self.domain_id[index]

        return {
            'image': image,
            'label': label,
            'domain': domain
        }

    def __len__(self):
        return self.len
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


class EEG(Dataset):
    def __init__(self, dataset, status):
        super(EEG, self).__init__()

        _status = ['train', 'val', 'test']
        if status not in _status:
            raise ValueError(f'Expected \'status\' in {_status}, but got {status}')

        X_train = dataset["samples"]
        Y_train = dataset["labels"]
        # self.kernel_size = 3
        # self.padding_length = 2
        # self.mean_kernel = torch.ones(1, 1, self.kernel_size) / self.kernel_size
        # self.dilated_conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size, dilation=self.padding_length)
        # self.dilated_conv.weight.data.copy_(self.mean_kernel)

        if len(X_train.shape) < 3:  # 需要三维的向量
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray): # 转换为torch
            X_train = torch.from_numpy(X_train)
            Y_train = torch.from_numpy(Y_train).long()  # 将数据类型转换为64位整数

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)  # 用于指定新的维度顺序

        self.x_data = X_train
        self.y_data = Y_train

        self.num_channels = X_train.shape[1]

        # if normalize:
        #     # Assume datashape: num_samples, num_channels, seq_length
        #     data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
        #     data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
        #     data_transform = transforms.Normalize(mean=data_mean, std=data_std)
        #     self.transform = data_transform
        # else:
        #     self.transform = None

        self.len = X_train.shape[0] # 时序样本数
        self.domain_id = [0] * len(self.y_data)

    def __getitem__(self, index):
        # if self.transform is not None:
        #     output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
        #     self.x_data[index] = output.view(self.x_data[index].shape)
        label = self.y_data[index].long()
        image = self.x_data[index]
        # image = series2img(self, self.x_data[index]).float()
        domain = self.domain_id[index]
        # return self.x_data[index].float(), self.y_data[index].long()


        return {
            'image': image,
            'label': label,
            'domain': domain
        }


    def __len__(self):
        return self.len


def series2img(self, s):

    # tmp2 = []
    # tmp1 = []
    # tmp4 = []
    # tmp5 = []
    # for i in range(0, len(tmp3)):
    #     if(i % 2 == 0):
    #         tmp2.append(tmp3[i])
    #         tmp2.append(tmp3[i])
    # for i in range(0, len(tmp3)):
    #     if (i % 3 == 0):
    #         tmp1.append(tmp3[i])
    #         tmp1.append(tmp3[i])
    #         tmp1.append(tmp3[i])
    # for i in range(0, len(tmp3)):
    #     if(i % 2 == 1):
    #         tmp4.append(tmp3[i])
    #         tmp4.append(tmp3[i])
    # for i in range(0, len(tmp3)):
    #     if (i % 3 == 2):
    #         tmp5.append(tmp3[i])
    #         tmp5.append(tmp3[i])
    #         tmp5.append(tmp3[i])
    # print(len(tmp1),len(tmp2),len(tmp4),len(tmp5))
    # print(np.concatenate(np.array((float_list)tmp1),np.array((float_list)tmp2)))
    # slen = len(s[0])
    # tmp3 = s
    # tmp2 = torch.repeat_interleave(s[:, ::2], repeats=2)  # 每隔2个元素重复一次
    # tmp1 = torch.repeat_interleave(s[:, ::3], repeats=3)  # 每隔3个元素重复两次
    # tmp4 = torch.repeat_interleave(s[:, 1::2], repeats=2)  # 从第二个元素开始每隔2个元素重复一次
    # tmp5 = torch.repeat_interleave(s[:, 2::3], repeats=3)  # 从第三个元素开始每隔3个元素重复两次
    # tmp2 = tmp2.reshape(1, slen)
    # tmp1 = tmp1.reshape(1, slen)
    # tmp4 = tmp4.reshape(1, slen)
    # tmp5 = tmp5.reshape(1, slen)
    # lay1 = torch.cat((tmp1, tmp2, tmp3, tmp4, tmp5), dim=0)
    #
    # # tmp1 = np.array(tmp1)
    # # tmp2 = np.vstack((tmp1, np.array(tmp2)))
    # # tmp3 = np.vstack((tmp2, tmp3))
    # # tmp4 = np.vstack((tmp3,  np.array(tmp4)))
    # # lay1 = np.vstack((tmp4,  np.array(tmp5)))
    # # print(tmp)
    #
    #
    # mean = 0
    # stddev = 1
    # noise1 = torch.randn(lay1.shape)  # 生成形状为 (5, 3000) 的随机噪声
    # noise1 = noise1 * stddev + mean
    # noise2 = torch.randn(lay1.shape)
    # noise2 = noise2 * stddev + mean
    # lay2 = lay1 + noise1
    # lay3 = lay1 + noise2
    # image_lay = torch.stack((lay1, lay2, lay3), dim=0)
    # return image_lay


    # print(noise2)
    # print(noise1)
    # lay = np.dstack((lay1,lay2,lay3))
    # print(lay.shape)
    # min_val = np.min(lay)
    # max_val = np.max(lay)
    # normalized_lay = (lay - min_val) / (max_val - min_val)
    # image_lay = (normalized_lay * 255).astype(np.uint8)
    # Shift lay1
    slen = len(s[0])
    channel = len(s)
    expand = s[:, -1].reshape(channel, -1)
    s = torch.cat([s, expand, expand], dim=1)
    tmp3 = s
    tmp2 = torch.repeat_interleave(s[:, ::2], repeats=2)  # 每隔2个元素重复一次
    tmp1 = torch.repeat_interleave(s[:, ::3], repeats=3)  # 每隔3个元素重复两次
    tmp4 = torch.repeat_interleave(s[:, 1::2], repeats=2)  # 从第二个元素开始每隔2个元素重复一次
    tmp5 = torch.repeat_interleave(s[:, 2::3], repeats=3)  # 从第三个元素开始每隔3个元素重复两次
    tmp3 = tmp3.reshape(channel, -1)[:, :slen]
    tmp2 = tmp2.reshape(channel, -1)[:, :slen]
    tmp1 = tmp1.reshape(channel, -1)[:, :slen]
    tmp4 = tmp4.reshape(channel, -1)[:, :slen]
    tmp5 = tmp5.reshape(channel, -1)[:, :slen]
    lay1 = torch.cat((tmp1, tmp2, tmp3, tmp4, tmp5), dim=0)


    # Augment lay2
    mean = 0
    stddev = 1
    noise1 = torch.randn(lay1.shape)  # 生成形状为 (5, 3000) 的随机噪声
    noise1 = noise1 * stddev + mean
    lay2 = lay1 + noise1

    # multiscale lay3
    pad_layer = nn.ConstantPad1d(self.padding_length, 0)
    tmp1 = self.dilated_conv(pad_layer(tmp1))
    tmp2 = self.dilated_conv(pad_layer(tmp2))
    tmp3 = self.dilated_conv(pad_layer(tmp3))
    tmp4 = self.dilated_conv(pad_layer(tmp4))
    tmp5 = self.dilated_conv(pad_layer(tmp5))
    lay3 = torch.cat((tmp1, tmp2, tmp3, tmp4, tmp5), dim=0)
    lay3 = lay3.detach()

    image_lay = torch.stack((lay1, lay2, lay3), dim=0)
    return image_lay

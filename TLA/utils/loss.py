# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_funcs import grl_hook, entropy_func


class WeightBCE(nn.Module):
    def __init__(self, epsilon: float = 1e-8) -> None:
        super(WeightBCE, self).__init__()
        self.epsilon = epsilon

    def forward(self, x: Tensor, label: Tensor, weight: Tensor) -> Tensor:
        """
        :param x: [N, 1]
        :param label: [N, 1]
        :param weight: [N, 1]
        """
        label = label.float()
        cross_entropy = - label * torch.log(x + self.epsilon) - (1 - label) * torch.log(1 - x + self.epsilon)
        return torch.sum(cross_entropy * weight.float()) / 2.


def d_align_uda(softmax_output: Tensor, features: Tensor = None, d_net=None,
                coeff: float = None, ent: bool = False):
    loss_func = WeightBCE()

    d_input = softmax_output if features is None else features
    d_output = d_net(d_input, coeff=coeff)
    d_output = torch.sigmoid(d_output)

    batch_size = softmax_output.size(0) // 2
    labels = torch.tensor([[1]] * batch_size + [[0]] * batch_size).long().cuda()  # 2N x 1

    if ent:
        x = softmax_output
        entropy = entropy_func(x)
        entropy.register_hook(grl_hook(coeff))
        entropy = torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[batch_size:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[:batch_size] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

    else:
        weight = torch.ones_like(labels).float() / batch_size

    loss_alg = loss_func.forward(d_output, labels, weight.view(-1, 1))

    return loss_alg

def d_taskalign_uda(features_all : Tensor, label_all : Tensor):
    average_distance = find_and_compute_cosine_similarity(features_all, label_all, max_pairs_per_label=2)
    return average_distance

def find_and_compute_cosine_similarity(vectors, labels, max_pairs_per_label=2):
    # 初始化一个空列表来存储相似度
    similarities = []

    # 找到每个标签对应的样本索引
    label_indices = {}
    for i, label in enumerate(labels):
        if label.item() not in label_indices:
            label_indices[label.item()] = []
        label_indices[label.item()].append(i)

    # 对每个标签找到最多两对样本
    for label, indices in label_indices.items():
        pairs_to_select = min(max_pairs_per_label, len(indices))

        # 从标签对应的样本中选择前 pairs_to_select 个
        selected_indices = indices[:pairs_to_select]

        # 计数器，确保每个标签最多选择两个样本
        count = 0

        # 计算选定样本之间的余弦相似度
        for i in selected_indices:
            for j in selected_indices:
                if i != j:
                    similarity = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0))
                    similarities.append(similarity.item())
                    count += 1

                    # 当达到每个标签最多选择两个样本时，退出内部循环
                    if count >= max_pairs_per_label:
                        break

            # 当达到每个标签最多选择两个样本时，退出外部循环
            if count >= max_pairs_per_label:
                break

    # 计算相似度的平均值
    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        return average_similarity
    else:
        return 0


def d_align_msda(softmax_output: Tensor, features: Tensor = None, d_net=None,
                 coeff: float = None, ent: bool = False, batchsizes: list = []):
    d_input = softmax_output if features is None else features
    d_output = d_net(d_input, coeff=coeff)

    labels = torch.cat(
        (torch.tensor([1] * batchsizes[0]).long(), torch.tensor([0] * batchsizes[1]).long()), 0
    ).cuda()  # [B_S + B_T]

    if ent:
        x = softmax_output
        entropy = entropy_func(x)
        entropy.register_hook(grl_hook(coeff))
        entropy = torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[batchsizes[0]:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[:batchsizes[0]] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

    else:
        weight = torch.ones_like(labels).float() / softmax_output.shape[0]

    loss_ce = nn.CrossEntropyLoss(reduction='none')(d_output, labels)
    loss_alg = torch.sum(weight * loss_ce)

    return loss_alg


# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, output1, output2, target):
#         euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
#         loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
#                                       (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive

class MMD(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def _guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        # number of used samples
        batch_size = int(source.size()[0])
        kernels = self._guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                        fix_sigma=self.fix_sigma)

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def probability_triplet_loss(anchor, positive, negative, margin=1.0):

    # 计算 anchor 与 positive 之间的欧氏距离的平方
    pos_dist = F.pairwise_distance(anchor, positive, p=2).pow(2)
    # 计算 anchor 与 negative 之间的欧氏距离的平方
    neg_dist = F.pairwise_distance(anchor, negative, p=2).pow(2)
    # 计算 pos_dist 和 neg_dist 的最小值和最大值
    min_pos_dist = pos_dist.min()
    max_pos_dist = pos_dist.max()

    min_neg_dist = neg_dist.min()
    max_neg_dist = neg_dist.max()

    # 对 pos_dist 进行归一化
    normalized_pos_dist = (pos_dist - min_pos_dist) / (max_pos_dist - min_pos_dist)

    # 对 neg_dist 进行归一化
    normalized_neg_dist = (neg_dist - min_neg_dist) / (max_neg_dist - min_neg_dist)
    # 转换为概率形式
    pos_prob = F.softmax(normalized_pos_dist, dim=0)
    neg_prob = F.softmax(normalized_neg_dist, dim=0)
    # 交叉熵损失
    loss = -torch.log(pos_prob / (pos_prob + neg_prob))
    # 返回平均损失
    return torch.mean(loss)

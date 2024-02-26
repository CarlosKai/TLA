# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.torch_funcs import entropy_func
import utils.loss as loss
from .base_trainer import *
from models import *
from utils.utils import get_coeff
import numpy as np
from utils import GradCAM


__all__ = ['ToAlign']


class ToAlign(BaseTrainer):
    def __init__(self, cfg):
        super(ToAlign, self).__init__(cfg)
        # self.criterion = loss.ContrastiveLoss(margin=1.0)
        self.batch = self.cfg.TRAIN.BATCH_SIZE_SOURCE
        self.criterion = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

    def build_base_models(self):
        basenet_name = self.cfg.MODEL.BASENET
        kwargs = {
            'pretrained': self.cfg.MODEL.PRETRAIN,
            'num_classes': self.cfg.DATASET.NUM_CLASSES,
            'toalign': self.cfg.METHOD.TOALIGN,
        }

        basenet = eval(basenet_name)(**kwargs).cuda()

        return basenet

    def build_models(self):
        logging.info(f'--> building models: {self.cfg.MODEL.BASENET}')
        # backbone
        self.base_net = self.build_base_models()
        # discriminator
        self.d_net = eval(self.cfg.MODEL.DNET)(
            in_feature=self.cfg.DATASET.NUM_CLASSES,
            # in_feature=2048,
            hidden_size=self.cfg.MODEL.D_HIDDEN_SIZE,
            out_feature=self.cfg.MODEL.D_OUTDIM
        ).cuda()


        self.learnable_matrix = nn.Parameter(torch.ones(self.cfg.DATASET.CHANNEL, self.cfg.DATASET.NUM_CLASSES)).cuda()

        # 创建五个不同尺度的 Conv1d
        # ==========================================================================================================================================================
        self.conv1d_list = []
        in_channels = 1  # 假设输入数据具有1个通道
        out_channels = 1  # 输出通道数
        kernel_sizes = [3, 5, 7, 9]
        paddings = [1, 2, 3, 4]
        multiscale = 4
        for scale in range(0, multiscale):
            conv1d = nn.Conv1d(in_channels, out_channels, kernel_sizes[scale],
                               padding=paddings[scale])  # 设置padding的起点终点为0，也暗含了时间序列起始和结束的信息
            self.conv1d_list.append(conv1d.to('cuda'))
        self.conv1d_list[0].weight.data.normal_(0, 0.01)
        self.conv1d_list[1].weight.data.normal_(0, 0.01)
        self.conv1d_list[2].weight.data.normal_(0, 0.01)
        self.conv1d_list[3].weight.data.normal_(0, 0.01)
        # ==========================================================================================================================================================


        self.registed_models = {'base_net': self.base_net, 'd_net': self.d_net, 'tcn1': self.conv1d_list[0],
                                'tcn2': self.conv1d_list[1], 'tcn3': self.conv1d_list[2], 'tcn4': self.conv1d_list[3]}
        # , 'task_metric': self.learnable_matrix
        self.model_parameters()     # 计算模型的大小
        parameter_list = self.base_net.get_parameters() + self.d_net.get_parameters()
        self.build_optim(parameter_list)

    def time_map_net(self, input_data):
        

        

    def test_func(self, loader, model):
        with torch.no_grad():
            iter_test = iter(loader)
            print_freq = max(len(loader) // 5, self.cfg.TRAIN.PRINT_FREQ)
            accs = AverageMeter()
            for i in range(len(loader)):
                if i % print_freq == print_freq - 1:
                    logging.info('    I:  {}/{} | acc: {:.3f}'.format(i, len(loader), accs.avg))
                data = iter_test.__next__()
                pre_inputs0, labels = data['image'].cuda(), data['label'].cuda()

                if self.cfg.METHOD.MULTI_TASK:
                    pre_inputs = torch.matmul(pre_inputs0.permute(0, 2, 1), self.learnable_matrix).permute(0, 2, 1).reshape(len(pre_inputs0), 1, -1).clone()

                else:
                    pre_inputs = pre_inputs0.reshape(len(pre_inputs0), 1, -1).clone()


                if self.cfg.METHOD.TIMEMAP:
                    inputs = self.time_map_net(pre_inputs)
                else:
                    inputs = pre_inputs

                outputs_all = model(inputs)  # [f, y, ...]
                outputs = outputs_all[1]

                acc = accuracy(outputs, labels)[0]
                maxk = 1
                _, pred = outputs.topk(maxk, 1, True, True)
                pred = pred.t()
                macro_f1 = f1_score(pred.reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy(), average='macro')

                accs.update(acc.item(), labels.size(0))

        return accs.avg, macro_f1

    def one_step(self, data_src, data_tar):
        pre_inputs_src0, labels_src = data_src['image'].cuda(), data_src['label'].cuda()
        pre_inputs_tar0, labels_tar = data_tar['image'].cuda(), data_tar['label'].cuda()

        if self.cfg.METHOD.MULTI_TASK:
            pre_inputs_src = torch.matmul(pre_inputs_src0.permute(0, 2, 1), self.learnable_matrix).permute(0, 2, 1).reshape(self.batch, 1, -1).clone()
            pre_inputs_tar = torch.matmul(pre_inputs_tar0.permute(0, 2, 1), self.learnable_matrix).permute(0, 2, 1).reshape(self.batch, 1, -1).clone()
        else:
            pre_inputs_src = pre_inputs_src0.reshape(self.batch, 1, -1).clone()
            pre_inputs_tar = pre_inputs_tar0.reshape(self.batch, 1, -1).clone()

        if self.cfg.METHOD.TIMEMAP:
            inputs_src = self.time_map_net(pre_inputs_src)
            inputs_tar = self.time_map_net(pre_inputs_tar)
        else:
            inputs_src = pre_inputs_src
            inputs_tar = pre_inputs_tar

        # --------- classification --------------
        outputs_all_src = self.base_net(inputs_src)  # 3个tuple [f, y, z]
        assert len(outputs_all_src) == 2, \
            f'Expected return with size 2, but got {len(outputs_all_src)}'
        loss_cls_src = F.cross_entropy(outputs_all_src[1], labels_src)      # 源域分类损失
        # focals_src = outputs_all_src[-1]



        outputs_all_tar = self.base_net(inputs_tar)  # [f, y, z]    # 目标域的特征
        outputs_all_src = self.base_net(inputs_src)

        # --------- contrastive learning --------------
        




        # --------- alignment --------------
        assert len(outputs_all_src) == 2 and len(outputs_all_tar) == 2, \
            f'Expected return with size 2, but got {len(outputs_all_src)}'


        # classificaiton loss
        loss_cls_tar = F.cross_entropy(outputs_all_tar[1].data, labels_tar)  # 计算目标域的分类损失，只用于观察，不用于训练模型

        # domain alignment  域判别损失
        loss_alg_task = torch.tensor(0).cuda()

        if self.cfg.METHOD.TOALIGN:
            outputs_all_src_task = self.base_net(inputs_src, toalign=True, labels=labels_src)  # [f_p, y_p, z_p]
            outputs_all_tar_task = self.base_net(inputs_tar, taskalign=True)
            logits_all_task = torch.cat((outputs_all_src_task[1], outputs_all_tar_task[1]), dim=0)
            softmax_all_task = nn.Softmax(dim=1)(logits_all_task)
            loss_alg_task = loss.d_align_uda(
                softmax_output=softmax_all_task, d_net=self.d_net,
                coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
            )
            logits_all = torch.cat((outputs_all_src_task[1], outputs_all_tar[1]), dim=0)
            softmax_all = nn.Softmax(dim=1)(logits_all)
            loss_alg = loss.d_align_uda(
                softmax_output=softmax_all, d_net=self.d_net,
                coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
            )
        else:
            logits_all = torch.cat((outputs_all_src[1], outputs_all_tar[1]), dim=0)
            softmax_all = nn.Softmax(dim=1)(logits_all)
            loss_alg = loss.d_align_uda(
                softmax_output=softmax_all, d_net=self.d_net,
                coeff=get_coeff(self.ite, max_iter=self.cfg.TRAIN.TTL_ITE), ent=self.cfg.METHOD.ENT
            )


        loss_alg_task = loss_alg + loss_alg_task * 0.1

        loss_ttl = loss_cls_src + loss_alg_task * self.cfg.METHOD.W_ALG  + loss_contra * self.cfg.METHOD.W_CONTRA
        # loss_ttl = loss_cls_src + (loss_alg + loss_alg_task) * self.cfg.METHOD.W_ALG + loss_contra * self.cfg.METHOD.W_CONTRA


        # update
        self.step(loss_ttl)

        # display
        if self.ite % self.cfg.TRAIN.PRINT_FREQ == 0:
            self.display([
                f'l_cls_src: {loss_cls_src.item():.3f}',
                f'l_cls_tar: {loss_cls_tar.item():.3f}',
                f'l_alg: {loss_alg.item():.3f}',
                f'l_contra: {loss_contra.item():.3f}',
                f'l_ttl: {loss_ttl.item():.3f}',
                f'best_acc: {self.best_acc:.3f}',
                f'best_macro_f1: {self.best_macro_f1:.3f}',
            ])
            # tensorboard
            # self.update_tb({
            #     'l_cls_src': loss_cls_src.item(),
            #     'l_cls_tar': loss_cls_tar.item(),
            #     'l_alg': loss_alg.item(),
            #     'l_hda': loss_hda.item(),
            #     'l_ttl': loss_ttl.item(),
            #     'ent_tar': ent_tar.item(),
            # })
            self.update_tb({
                'l_cls_src': loss_cls_src.item(),
                'l_cls_tar': loss_cls_tar.item(),
                'l_alg': loss_alg.item(),
                'l_contra': loss_contra.item(),
                'l_ttl': loss_ttl.item(),
                'l_alg_total':loss_alg_task.item(),
            })

    def save_model(self, is_best=False, snap=False):
        # data_dict = {
        #     'optimizer': self.optimizer.state_dict(),
        #     'ite': self.ite,
        #     'best_acc': self.best_acc,
        #     'best_macro_f1': self.best_macro_f1,
        #     'learnable_matrix': self.learnable_matrix
        # }
        # for k, v in self.registed_models.items():
        #     data_dict.update({k: v.state_dict()})
        # save_model(self.cfg.TRAIN.OUTPUT_CKPT, data_dict=data_dict, ite=self.ite, is_best= is_best, snap=snap)
        pass


    def resume_from_ckpt(self):
        last_ckpt = os.path.join(self.cfg.TRAIN.OUTPUT_CKPT, 'models-last.pt')
        if os.path.exists(last_ckpt):
            ckpt = torch.load(last_ckpt)
            for k, v in self.registed_models.items():
                v.load_state_dict(ckpt[k])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_ite = ckpt['ite']
            self.best_acc = ckpt['best_acc']
            self.best_macro_f1 = ckpt['best_macro_f1']
            self.learnable_matrix = ckpt['learnable_matrix']
            logging.info(f'> loading ckpt from {last_ckpt} | ite: {self.start_ite} | best_acc: {self.best_acc:.3f} | best_macro_f1: {self.best_macro_f1:.3f}')
        else:
            logging.info('--> training from scratch')

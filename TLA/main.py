# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import json
import argparse
import torch.backends.cudnn as cudnn
from configs.defaults import get_default_and_update_cfg
from utils.utils import create_logger, set_seed
from trainer import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flag = 6
try_data = "EEG"
# try_data = "HAR"
# try_data = "WISDM"
# try_data = "HHAR"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',      default='configs/uda_' + try_data + '_toalign.yaml', type=str)
    parser.add_argument('--seed',   default=123, type=int)
    parser.add_argument('--source', default= try_data + "_12", nargs='+', help='source domain names')
    parser.add_argument('--target', default= try_data + "_5", nargs='+', help='target domain names')
    parser.add_argument('--output_root', default="exp_" +  try_data, type=str, help='output root path')
    parser.add_argument('--output_dir',  default="exp_" +  try_data + "12_5_output_dir", type=str, help='output path, subdir under output_root')
    parser.add_argument('--data_root',   default="./data/" + try_data, type=str, help='path to dataset root')
    parser.add_argument('--opts',   default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert os.path.isfile(args.cfg), 'cfg file: {} not found'.format(args.cfg)

    return args


def main():
    args = parse_args()

    
    # EEG
    if flag==1:
        EEG_configurations = [
            (0, 11),
            (2, 5),
            (12, 5),
            (7, 18),
            (16, 1),
            (9, 14),
            (4, 12),
            (10, 7),
            (6, 3),
            (8, 10),
        ]
        for i, j in EEG_configurations:
            args.source = try_data + "_" + str(i)
            args.target = try_data + "_" + str(j)
            args.output_dir = "expcheer_" + try_data + str(i) + "_" + str(j) + "_output_dir"
            cfg = get_default_and_update_cfg(args)
            set_seed(cfg.SEED)
            cudnn.deterministic = True
            cudnn.benchmark = False
            logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
            logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))    # 记录命令行参数的信息
            logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))                   # 记录配置参数的信息。
            trainer = eval(cfg.TRAINER)(cfg)
            trainer.train()

    # HAR
    if flag == 2:
        HAR_configurations = [(3, 12), (1, 22), (2, 22), (22, 30), (21, 27), (17, 21), (11, 29), (1, 20), (11, 19), (1, 20), (17, 28), (14, 15), (7, 17), (20, 30), (19, 22), (1, 4), (17, 26), (23, 29), (13, 30), (5, 6), (20, 26), (20, 29), (6, 29), (14, 28), (5, 13), (22, 26), (2, 27), (4, 29), (4, 22), (19, 26), (1, 9), (17, 19), (3, 5), (24, 30), (4, 30), (3, 18), (2, 14), (9, 19), (4, 22), (3, 6), (2, 5), (5, 14), (19, 21), (3, 23), (17, 30), (17, 25), (11, 19), (17, 21), (1, 14), (24, 29), (1, 13), (5, 18), (5, 6), (15, 28), (13, 23), (15, 25), (25, 29), (6, 15), (10, 14), (26, 29), (2, 6), (13, 27), (21, 27), (21, 22), (2, 9), (6, 12), (6, 14), (8, 30), (1, 3), (6, 13), (17, 27), (5, 18), (18, 19), (9, 28), (5, 6), (19, 28), (9, 15), (9, 14), (10, 25), (10, 28), (9, 10), (23, 26), (2, 20), (10, 22), (5, 8), (5, 26), (17, 22), (7, 30), (14, 21), (9, 13), (5, 12), (5, 6), (17, 27), (22, 27), (15, 20), (5, 23), (4, 8), (8, 10), (9, 10), (2, 14)]


        for i, j in HAR_configurations:
            args.source = try_data + "_" + str(i)
            args.target = try_data + "_" + str(j)
            args.output_dir = "exp_" + try_data + str(i) + "_" + str(j) + "_output_dir"
            cfg = get_default_and_update_cfg(args)
            set_seed(cfg.SEED)
            cudnn.deterministic = True
            cudnn.benchmark = False
            logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
            logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))    # 记录命令行参数的信息
            logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))                   # 记录配置参数的信息。
            trainer = eval(cfg.TRAINER)(cfg)
            trainer.train()

    # WISDM
    if flag==3:
        WISDM_configurations = [(3, 13), (5, 10), (12, 25), (23, 25), (9, 17), (23, 34), (2, 11), (4, 30), (6, 11), (13, 26), (7, 19), (16, 31), (20, 31), (3, 24), (5, 21), (14, 24), (14, 33), (11, 16), (6, 13), (7, 12), (7, 30), (12, 20), (6, 34), (5, 14), (3, 26), (0, 9), (4, 25), (0, 30), (20, 26), (14, 19), (9, 14), (8, 27), (13, 23), (0, 32), (12, 24), (23, 33), (1, 5), (1, 14), (27, 33), (13, 16), (10, 26), (2, 22), (0, 34), (6, 22), (20, 21), (12, 17), (3, 14), (6, 31), (26, 34), (12, 26), (3, 23), (8, 22), (1, 16), (2, 15), (7, 11), (2, 33), (26, 27), (4, 15), (23, 28), (12, 28), (8, 24), (2, 17), (2, 26), (24, 32), (5, 6), (14, 18), (12, 30), (2, 10), (17, 29), (1, 20), (0, 31), (2, 28), (11, 31), (15, 22), (6, 19), (12, 14), (3, 11), (18, 27), (12, 23), (15, 24), (6, 21), (7, 20), (1, 34), (7, 29), (10, 18), (2, 14), (9, 29), (6, 14), (16, 25), (7, 31), (10, 11), (8, 23), (2, 16), (11, 28), (24, 31), (25, 30), (22, 24), (14, 20), (22, 33), (5, 26)]
        # WISDM_configurations = [(3,13),(12,25),(26,34),(11,28),(6,14)]


        for i, j in WISDM_configurations:
            args.source = try_data + "_" + str(i)
            args.target = try_data + "_" + str(j)
            args.output_dir = "expcheer_" + try_data + str(i) + "_" + str(j) + "_output_dir"
            cfg = get_default_and_update_cfg(args)
            set_seed(cfg.SEED)
            cudnn.deterministic = True
            cudnn.benchmark = False
            logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
            logger.info('======================= args =======================\n' + json.dumps(vars(args),
                                                                                              indent=4))  # 记录命令行参数的信息
            logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))  # 记录配置参数的信息。
            trainer = eval(cfg.TRAINER)(cfg)
            trainer.train()

    if flag==4:
        # HHAR_configurations = [
        #     (0,4),
        #     (4,8),
        #     (8,2),
        #     (2,3),
        #     (3,8),

        # ]
        HHAR_configurations = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
            (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
            (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
            (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
            (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8),
            (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 8),
            (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 8),
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
        ]
        for i, j in HHAR_configurations:
            args.source = try_data + "_" + str(i)
            args.target = try_data + "_" + str(j)
            args.output_dir = "expcheer_" + try_data + str(i) + "_" + str(j) + "_output_dir"
            cfg = get_default_and_update_cfg(args)
            set_seed(cfg.SEED)
            cudnn.deterministic = True
            cudnn.benchmark = False
            logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
            logger.info('======================= args =======================\n' + json.dumps(vars(args),
                                                                                              indent=4))  # 记录命令行参数的信息
            logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))  # 记录配置参数的信息。
            trainer = eval(cfg.TRAINER)(cfg)
            trainer.train()

    if flag==5:
      for i in range(0, 9):
        for j in range(0, 9):
            if(i == j):
                continue;
            else:
                args.source = try_data + "_" + str(i)
                args.target = try_data + "_" + str(j)
                args.output_dir = "exp_" + try_data + str(i) + "_" + str(j) + "_output_dir"
                cfg = get_default_and_update_cfg(args)
    
                # cfg.freeze()
                # seed
                set_seed(cfg.SEED)
                cudnn.deterministic = True
                cudnn.benchmark = False
    
                # logger
                logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
                logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))    # 记录命令行参数的信息
                logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))                   # 记录配置参数的信息。
    
                trainer = eval(cfg.TRAINER)(cfg)    # 相当于在代码中写下了Trainer(cfg)，从而创建了一个名为trainer的对象。
                trainer.train()

        # EEG_sensitive_analysis
    if flag == 6:
        EEG_configurations = [
            (12, 5),
            (6, 3),
            (7, 18),
            (0, 11),
            (9, 14),
        ]
        contra = [0.1, 0.3, 0.5, 1, 5, 10]
        align = [0.1, 0.3, 0.5, 1, 5, 10]
        for k in contra:
            for i, j in EEG_configurations:
                args.source = try_data + "_" + str(i)
                args.target = try_data + "_" + str(j)
                args.output_dir = "exp_contra" + str(k) + "_" + try_data + "_" + str(i) + "_" + str(j) + "_output_dir"
                cfg = get_default_and_update_cfg(args)
                cfg.METHOD.W_CONTRA = k
                set_seed(cfg.SEED)
                cudnn.deterministic = True
                cudnn.benchmark = False
                logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
                logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))  # 记录命令行参数的信息
                logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))  # 记录配置参数的信息。
                trainer = eval(cfg.TRAINER)(cfg)
                trainer.train()
        for k in align:
            for i, j in EEG_configurations:
                args.source = try_data + "_" + str(i)
                args.target = try_data + "_" + str(j)
                args.output_dir = "exp_align" + str(k) + "_" + try_data + "_" + str(i) + "_" + str(j) + "_output_dir"
                cfg = get_default_and_update_cfg(args)
                cfg.METHOD.W_ALG = k
                set_seed(cfg.SEED)
                cudnn.deterministic = True
                cudnn.benchmark = False
                logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
                logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))  # 记录命令行参数的信息
                logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))  # 记录配置参数的信息。
                trainer = eval(cfg.TRAINER)(cfg)
                trainer.train()

    # ablation
    if flag == 7:
        # HHAR_configurations = [
        #     (0, 8),
        #     (5, 0),
        #     (7, 3),
        #     (3, 2),
        #     (4, 2),
        # ]
        # HHAR_configurations = [
        #     (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        #     (1, 0), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
        #     (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        #     (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        #     (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 8),
        #     (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 8),
        #     (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 8),
        #     (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 8),
        #     (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7),
        # ]
        HHAR_configurations = [
            (0,4),
            (2,1),
            (3,0),
            (4,8),
            (7,2),
        ]

        for k in range(0,5):
            for i, j in HHAR_configurations:
                args.source = try_data + "_" + str(i)
                args.target = try_data + "_" + str(j)
                args.output_dir = "exp_ablation" + str(k) + "_" + try_data + "_" + str(i) + "_" + str(j) + "_output_dir"
                cfg = get_default_and_update_cfg(args)
                if k==0:
                    cfg.METHOD.CONTRA = False
                    cfg.METHOD.MULTI_TASK = False
                    cfg.METHOD.TOALIGN = False
                if k==1:
                    cfg.METHOD.CONTRA = False
                    cfg.METHOD.MULTI_TASK = True
                    cfg.METHOD.TOALIGN = False
                if k==2:
                    cfg.METHOD.CONTRA = True
                    cfg.METHOD.MULTI_TASK = True
                    cfg.METHOD.TOALIGN = False
                if k==3:
                    cfg.METHOD.CONTRA = True
                    cfg.METHOD.MULTI_TASK = True
                    cfg.METHOD.TOALIGN = True
                if k==4:
                    cfg.METHOD.CONTRA = False
                    cfg.METHOD.MULTI_TASK = True
                    cfg.METHOD.TOALIGN = True
                set_seed(cfg.SEED)
                cudnn.deterministic = True
                cudnn.benchmark = False
                logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
                logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))  # 记录命令行参数的信息
                logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))  # 记录配置参数的信息。
                trainer = eval(cfg.TRAINER)(cfg)
                trainer.train()



    cfg = get_default_and_update_cfg(args)
    cfg.freeze()

    # seed
    set_seed(cfg.SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False

    # logger
    logger = create_logger('cfg.TRAIN.OUTPUT_LOG')
    logger.info('======================= args =======================\n' + json.dumps(vars(args), indent=4))    # 记录命令行参数的信息
    logger.info('======================= cfg =======================\n' + cfg.dump(indent=4))                   # 记录配置参数的信息。

    trainer = eval(cfg.TRAINER)(cfg)    # 相当于在代码中写下了Trainer(cfg)，从而创建了一个名为trainer的对象。
    trainer.train()




if __name__ == '__main__':
    main()

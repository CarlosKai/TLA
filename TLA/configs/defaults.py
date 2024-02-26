# --------------------------------------------------------
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os

from yacs.config import CfgNode as CN


_C = CN()
_C.SEED = 123
_C.WORKERS = 4
_C.TRAINER = 'Trainer'


# ================= task ====================
_C.TASK = CN()
_C.TASK.NAME = 'UDA'
_C.TASK.SSDA_SHOT = 1

# ================= training ====================
_C.TRAIN = CN()
_C.TRAIN.TEST_FREQ = 500
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.SAVE_FREQ = 5000
_C.TRAIN.TTL_ITE = 8000

_C.TRAIN.BATCH_SIZE_SOURCE = 36
_C.TRAIN.BATCH_SIZE_TARGET = 36
_C.TRAIN.BATCH_SIZE_TEST = 36
_C.TRAIN.LR = 0.001

_C.TRAIN.OUTPUT_ROOT = 'temp'
_C.TRAIN.OUTPUT_DIR = ''
_C.TRAIN.OUTPUT_LOG = 'log' # 保存日志
_C.TRAIN.OUTPUT_TB = 'tensorboard'  # 保存参数可视化
_C.TRAIN.OUTPUT_CKPT = 'ckpt'   # 保存模型目录
_C.TRAIN.OUTPUT_RESFILE = 'log.txt'

# ================= models ====================
_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 5e-4
_C.OPTIM.MOMENTUM = 0.9

# ================= models ====================
_C.MODEL = CN()
_C.MODEL.PRETRAIN = True
_C.MODEL.BASENET = 'resent50'
_C.MODEL.BASENET_DOMAIN_EBD = False  # for domain embedding for transformer
_C.MODEL.DNET = 'Discriminator'
_C.MODEL.D_INDIM = 0
_C.MODEL.D_OUTDIM = 1
_C.MODEL.D_HIDDEN_SIZE = 1024
_C.MODEL.D_WGAN_CLIP = 0.01
_C.MODEL.VIT_DPR = 0.1
_C.MODEL.VIT_USE_CLS_TOKEN = True
_C.MODEL.VIT_PRETRAIN_EXLD = []
# extra layer
_C.MODEL.EXT_LAYER = False
_C.MODEL.EXT_NUM_TOKENS = 100
_C.MODEL.EXT_NUM_LAYERS = 1
_C.MODEL.EXT_NUM_HEADS = 24
_C.MODEL.EXT_LR = 10.
_C.MODEL.EXT_DPR = 0.1
_C.MODEL.EXT_SKIP = True
_C.MODEL.EXT_FEATURE = 768

# ================= dataset ====================
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.NUM_CLASSES = 10
_C.DATASET.NAME = ''
_C.DATASET.SOURCE = []
_C.DATASET.TARGET = []
_C.DATASET.TRIM = 0
_C.DATASET.CHANNEL = 1

# ================= method ====================
_C.METHOD = CN()
_C.METHOD.W_ALG = 1.0
_C.METHOD.W_CONTRA = 0.5
_C.METHOD.ENT = False
_C.METHOD.CONTRA = False
_C.METHOD.MULTI_TASK = False
_C.METHOD.TOALIGN = False
_C.METHOD.TIMEMAP = True


def get_default_and_update_cfg(args):
    cfg = _C.clone()
    cfg.merge_from_file(args.cfg)
    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.SEED = args.seed

    if args.data_root:
        cfg.DATASET.ROOT = args.data_root

    # dataset maps
    maps = {
        'EEG': {
            'EEG_0': "0",
            'EEG_1': "1",
            'EEG_2': "2",
            'EEG_3': "3",
            'EEG_4': "4",
            'EEG_5': "5",
            'EEG_6': "6",
            'EEG_7': "7",
            'EEG_8': "8",
            'EEG_9': "9",
            'EEG_10': "10",
            'EEG_11': "11",
            'EEG_12': "12",
            'EEG_13': "13",
            'EEG_14': "14",
            'EEG_15': "15",
            'EEG_16': "16",
            'EEG_17': "17",
            'EEG_18': "18",
            'EEG_19': "19",
        },
        'HHAR': {
            'HHAR_0': "0",
            'HHAR_1': "1",
            'HHAR_2': "2",
            'HHAR_3': "3",
            'HHAR_4': "4",
            'HHAR_5': "5",
            'HHAR_6': "6",
            'HHAR_7': "7",
            'HHAR_8': "8",
        },
        'HAR': {
            'HAR_0': "0",
            'HAR_1': "1",
            'HAR_2': "2",
            'HAR_3': "3",
            'HAR_4': "4",
            'HAR_5': "5",
            'HAR_6': "6",
            'HAR_7': "7",
            'HAR_8': "8",
            'HAR_9': "9",
            'HAR_10': "10",
            'HAR_11': "11",
            'HAR_12': "12",
            'HAR_13': "13",
            'HAR_14': "14",
            'HAR_15': "15",
            'HAR_16': "16",
            'HAR_17': "17",
            'HAR_18': "18",
            'HAR_19': "19",
            'HAR_20': "20",
            'HAR_21': "21",
            'HAR_22': "22",
            'HAR_23': "23",
            'HAR_24': "24",
            'HAR_25': "25",
            'HAR_26': "26",
            'HAR_27': "27",
            'HAR_28': "28",
            'HAR_29': "29",
            'HAR_30': "30",
        },
        'WISDM': {
            'WISDM_0': "0",
            'WISDM_1': "1",
            'WISDM_2': "2",
            'WISDM_3': "3",
            'WISDM_4': "4",
            'WISDM_5': "5",
            'WISDM_6': "6",
            'WISDM_7': "7",
            'WISDM_8': "8",
            'WISDM_9': "9",
            'WISDM_10': "10",
            'WISDM_11': "11",
            'WISDM_12': "12",
            'WISDM_13': "13",
            'WISDM_14': "14",
            'WISDM_15': "15",
            'WISDM_16': "16",
            'WISDM_17': "17",
            'WISDM_18': "18",
            'WISDM_19': "19",
            'WISDM_20': "20",
            'WISDM_21': "21",
            'WISDM_22': "22",
            'WISDM_23': "23",
            'WISDM_24': "24",
            'WISDM_25': "25",
            'WISDM_26': "26",
            'WISDM_27': "27",
            'WISDM_28': "28",
            'WISDM_29': "29",
            'WISDM_30': "30",
            'WISDM_31': "31",
            'WISDM_32': "32",
            'WISDM_33': "33",
            'WISDM_34': "34",
            'WISDM_35': "35",
        }
    }

    cfg.DATASET.SOURCE = maps[cfg.DATASET.NAME][args.source]
    cfg.DATASET.TARGET = maps[cfg.DATASET.NAME][args.target]

    # class
    if  cfg.DATASET.NAME == 'EEG':
        cfg.DATASET.NUM_CLASSES = 5
    elif cfg.DATASET.NAME == 'HAR':
        cfg.DATASET.NUM_CLASSES = 6
    elif cfg.DATASET.NAME == 'HHAR':
        cfg.DATASET.NUM_CLASSES = 6
    elif cfg.DATASET.NAME == 'WISDM':
        cfg.DATASET.NUM_CLASSES = 6
    else:
        raise NotImplementedError(f'cfg.DATASET.NAME: {cfg.DATASET.NAME} not imeplemented')

    if  cfg.DATASET.NAME == 'EEG':
        cfg.DATASET.CHANNEL = 1
    elif cfg.DATASET.NAME == 'HAR':
        cfg.DATASET.CHANNEL = 9
    elif cfg.DATASET.NAME == 'HHAR':
        cfg.DATASET.CHANNEL = 3
    elif cfg.DATASET.NAME == 'WISDM':
        cfg.DATASET.CHANNEL = 3
    else:
        raise NotImplementedError(f'cfg.DATASET.CHANNEL: {cfg.DATASET.NAME} not imeplemented')

    # output
    if args.output_root:
        cfg.TRAIN.OUTPUT_ROOT = args.output_root
    if args.output_dir:
        cfg.TRAIN.OUTPUT_DIR = args.output_dir
    else:
        cfg.TRAIN.OUTPUT_DIR = '_'.join(cfg.DATASET.SOURCE) + '2' + '_'.join(cfg.DATASET.TARGET) + '_' + str(args.seed)

    #
    cfg.TRAIN.OUTPUT_CKPT = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'ckpt', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_LOG = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'log', cfg.TRAIN.OUTPUT_DIR)
    cfg.TRAIN.OUTPUT_TB = os.path.join(cfg.TRAIN.OUTPUT_ROOT, 'tensorboard', cfg.TRAIN.OUTPUT_DIR)
    os.makedirs(cfg.TRAIN.OUTPUT_CKPT, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_LOG, exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_TB, exist_ok=True)
    cfg.TRAIN.OUTPUT_RESFILE = os.path.join(cfg.TRAIN.OUTPUT_LOG, 'log.txt')

    return cfg


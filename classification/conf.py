# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - continual: do not reset, consider corruptions as a sequence; reset_each_shift: reset after each corruption
_C.SETTING = "continual"  # ["continual", "reset_each_shift", "non_stationary", "correlated", "non_stationary_correlated"]

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Output directory
_C.SAVE_DIR = "./output"

# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 1

# Deterministic experiments.
_C.DETERMINISM = False

# Optional description of a config
_C.DESC = ""

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10_c'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate (10000 for all samples in CIFAR-C)
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
# If num_ex is larger than 5000 all images (50,000) are loaded and then subsampled to num_ex
_C.CORRUPTION.NUM_EX = 10000

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3  # 0.0005 for vit_b_16

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999   # 0.9 for vit_b_16

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1/3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 1.0
_C.RMT.LAMBDA_CE_TRG = 1.0
_C.RMT.LAMBDA_CONT = 1.0
_C.RMT.NUM_SAMPLES_WARM_UP = 50000

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard" # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0
_C.ADACONTRAST.BETA = 1.0
_C.ADACONTRAST.ETA = 1.0

_C.ADACONTRAST.DIST_TYPE = "cosine" # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong" # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0   # [0, 1] possibility to reduce the number of source samples

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "domainnet126": "DomainNet-126",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "",
               "cifar100": "",
               "cifar100_c": "",
               }
    return os.path.join(root, mapping[dataset_name])


def get_num_classes(dataset_name):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100,  "cifar100_c": 100,
                                "imagenet": 1000, "imagenet_c": 1000, "imagenet_r": 200, "imagenet200": 200,
                                "domainnet126": 126,
                                }
    return dataset_name2num_classes[dataset_name]


def get_domain_sequence(ckpt_path):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt')
    domain = cfg.CKPT_PATH.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]

from yacs.config import CfgNode as CN
import argparse
import numpy as np

cfg = CN()
# cfg.dataset

cfg.train = CN()

# Distributed
cfg.train.distributed_addr = 'localhost'
cfg.train.distributed_port = '12845'
# Debugging
cfg.train.disable_progress_bar = False
cfg.train.disable_mixed_precision = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()


    return cfg

import sys
import json
import time
import os
import io
import pickle
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
import numpy as np

sys.path.append('/home/comp/19481691/occupy/reft/src')
sys.path.append('/home/comp/19481691/occupy/reft/src/models')

from ddp_model import LinearModel

class CheckpointMerger:
    def __init__(self, param_keys):
        self.param_keys = param_keys

    # @staticmethod


    def merge_sharded_ckpts(self, rank_files):
        merged_model_state_dict = {}
        merged_optimizer_state_dict = {'state': {}, 'param_groups': []}

        param_groups_added = False

        for idx, ckpt_path in enumerate(rank_files):
            checkpoint = self.load_checkpoint(ckpt_path)

            for key in checkpoint['model']:
                if key not in merged_model_state_dict:
                    merged_model_state_dict[key] = checkpoint['model'][key]

            if 'optimizer' in checkpoint and 'state' in checkpoint['optimizer']:
                merged_optimizer_state_dict['state'].update(checkpoint['optimizer']['state'])

            if not param_groups_added and 'optimizer' in checkpoint and 'param_groups' in checkpoint['optimizer']:
                merged_optimizer_state_dict['param_groups'] = checkpoint['optimizer']['param_groups']
                param_groups_added = True

        return {'model': merged_model_state_dict, 'optimizer': merged_optimizer_state_dict}

    def load_merged_ckpt_to_model(self, model, optimizer, rank_files):
        merged_ckpt = self.merge_sharded_ckpts(rank_files)
        prefixed_state_dict = {"module."+k: v for k, v in merged_ckpt['model'].items()}
        model.load_state_dict(prefixed_state_dict)
        optimizer.load_state_dict(merged_ckpt['optimizer'])
        
        

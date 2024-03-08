import pickle
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
import time
sys.path.append('/home/comp/19481691/occupy/reft/src')
sys.path.append('/home/comp/19481691/occupy/reft/src/models')
from ddp_model import LinearModel


class ByteXOR:
    def __init__(self, layer_ids, param_types):
        self.layer_ids = layer_ids
        self.param_types = param_types

    def xor_bytes(self, arrays_list):
        result_array = arrays_list[0]
        for array in arrays_list[1:]:
            result_array = np.bitwise_xor(result_array, array)
        return result_array

    def process_model(self, model):
        param_dict = {}
        for name, param in model.named_parameters():
            layer_id = int(name.split('.')[0][-1])
            param_type = 'weight' if 'weight' in name else 'bias'
            if layer_id in self.layer_ids and param_type in self.param_types:
                key = (param_type,)
                if key not in param_dict:
                    param_dict[key] = []
                param_dict[key].append(np.frombuffer(param.detach().numpy().tobytes(), dtype=np.int32))

        xor_results = {}
        start_time = time.time()
        total_bytes = 0
        for key, param_list in param_dict.items():
            total_bytes += sum(a.nbytes for a in param_list)
            xor_result = self.xor_bytes(param_list)
            xor_array = xor_result.view(np.float32)
            xor_tensor = torch.from_numpy(xor_array)
            xor_results[key] = xor_tensor
        end_time = time.time()

        elapsed_time = end_time - start_time
        speed = total_bytes / elapsed_time / (1024 ** 3)

        return xor_results, speed


    def recover_layer_params(self, model, parity_dict, lost_layer_id):
        recovered_params = {}
        for param_type in self.param_types:
            key = (lost_layer_id, param_type)
            if key in parity_dict:
                param_bytes_list = []
                for name, param in model.named_parameters():
                    layer_id = int(name.split('.')[0][-1])
                    if layer_id != lost_layer_id and param_type in name:
                        param_bytes_list.append(memoryview(param.detach().numpy()).tobytes())

                param_bytes_list.append(memoryview(parity_dict[key].numpy()).tobytes())
                recovered_xor_result = self.xor_bytes(param_bytes_list)
                recovered_array = np.frombuffer(recovered_xor_result, dtype=np.float32).reshape(parity_dict[key].shape)
                recovered_tensor = torch.tensor(recovered_array)
                recovered_params[param_type] = recovered_tensor

                # Update the model with the recovered parameter
                lost_layer = getattr(model, f'fc{lost_layer_id}')
                setattr(lost_layer, param_type, torch.nn.Parameter(recovered_tensor))

        return recovered_params

    def verify_xor(self, model, parity, excluded_layer_id):
        included_layer_ids = [id for id in self.layer_ids if id != excluded_layer_id]
        param_bytes_list = []
        for layer_id in included_layer_ids:
            layer = getattr(model, f'{self.layer_name}{layer_id}')
            param = getattr(layer, self.param_type)
            param_bytes_list.append(memoryview(param.detach().numpy()).tobytes())

        param_bytes_list.append(memoryview(parity.numpy()).tobytes())
        recovered_xor_result = self.xor_bytes(param_bytes_list)
        recovered_array = np.frombuffer(recovered_xor_result, dtype=np.float32).reshape(parity.shape)
        recovered_tensor = torch.tensor(recovered_array)

        original_layer = getattr(model, f'{self.layer_name}{excluded_layer_id}')
        original_param = getattr(original_layer, self.param_type)

        # Check if the recovered tensor is equal to the original parameter tensor
        is_equal = torch.all(torch.isclose(recovered_tensor, original_param.detach()))

        return is_equal, recovered_tensor
    

def load_checkpoint(filepath):
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

def merge_sharded_ckpts(rank_files, param_keys):
    merged_model_state_dict = {}
    merged_optimizer_state_dict = {'state': {}, 'param_groups': []}
    
    param_groups_added = False
    
    for idx, ckpt_path in enumerate(rank_files):
        checkpoint = load_checkpoint(ckpt_path)
        
        for key in checkpoint['model']:
            if key not in merged_model_state_dict:
                merged_model_state_dict[key] = checkpoint['model'][key]
                
        if 'optimizer' in checkpoint and 'state' in checkpoint['optimizer']:
            merged_optimizer_state_dict['state'].update(checkpoint['optimizer']['state'])
                
        if not param_groups_added and 'optimizer' in checkpoint and 'param_groups' in checkpoint['optimizer']:
            merged_optimizer_state_dict['param_groups'] = checkpoint['optimizer']['param_groups']
            param_groups_added = True
    
    return {'model': merged_model_state_dict, 'optimizer': merged_optimizer_state_dict}


param_keys = {
    "0": ["input_layer.weight", "input_layer.bias", "hidden_layers.0.weight", "hidden_layers.0.bias", "hidden_layers.4.weight", "hidden_layers.4.bias", "hidden_layers.8.weight", "hidden_layers.8.bias", "output_layer.weight", "output_layer.bias","hidden_layers.1.weight", "hidden_layers.1.bias", "hidden_layers.2.weight", "hidden_layers.2.bias", "hidden_layers.3.weight", "hidden_layers.3.bias"],
    "1": ["input_layer.weight", "input_layer.bias", "hidden_layers.1.weight", "hidden_layers.1.bias", "hidden_layers.5.weight", "hidden_layers.5.bias", "hidden_layers.9.weight", "hidden_layers.9.bias",  "output_layer.weight", "output_layer.bias","hidden_layers.4.weight", "hidden_layers.4.bias", "hidden_layers.6.weight", "hidden_layers.6.bias", "hidden_layers.7.weight", "hidden_layers.7.bias"],
    "2": ["input_layer.weight", "input_layer.bias", "hidden_layers.2.weight", "hidden_layers.2.bias", "hidden_layers.6.weight", "hidden_layers.6.bias", "hidden_layers.10.weight", "hidden_layers.10.bias",   "output_layer.weight", "output_layer.bias","hidden_layers.8.weight", "hidden_layers.8.bias", "hidden_layers.9.weight", "hidden_layers.9.bias", "hidden_layers.11.weight", "hidden_layers.11.bias"],
    "3": ["input_layer.weight", "input_layer.bias", "hidden_layers.3.weight", "hidden_layers.3.bias", "hidden_layers.7.weight", "hidden_layers.7.bias", "hidden_layers.11.weight", "hidden_layers.11.bias",  "output_layer.weight", "output_layer.bias","hidden_layers.10.weight", "hidden_layers.10.bias", "hidden_layers.0.weight", "hidden_layers.0.bias", "hidden_layers.5.weight", "hidden_layers.5.bias"]
}

rank_files = [
    '../benchmarks/checkpoints/sharded_checkpoint_rank_0_epoch_0.pth',
    '../benchmarks/checkpoints/sharded_checkpoint_rank_1_epoch_0.pth',
    '../benchmarks/checkpoints/sharded_checkpoint_rank_2_epoch_0.pth',
    '../benchmarks/checkpoints/sharded_checkpoint_rank_3_epoch_0.pth'
]
model = LinearModel()
optimizer = SGD(model.parameters(), lr=0.01)
merged_ckpt = merge_sharded_ckpts(rank_files, param_keys)
model.load_state_dict(merged_ckpt['model'])
optimizer.load_state_dict(merged_ckpt['optimizer'])


import torch
import numpy as np
import time
import torch.nn as nn
import os
import json
import sys
sys.path.append('/home/comp/19481691/occupy/reft/src')
sys.path.append('/home/comp/194816`91/occupy/reft/src/models')
from ddp_model import LinearModel
        
import numpy as np
import torch
import os
import time

class ByteXOR:
    def __init__(self, param_types, redundant_param_keys):
        self.param_types = param_types
        self.redundant_param_keys = redundant_param_keys

    def xor_bytes(self, arrays_list):
        result_array = arrays_list[0]
        for array in arrays_list[1:]:
            result_array = np.bitwise_xor(result_array, array)
        return result_array

    def process_model(self, model, rank):
        layer_ids = list(set(int(i.split('.')[1]) for i in self.redundant_param_keys[str(rank)] if 'hidden_layers' in i))
        param_dict = self._extract_parameters(model, layer_ids)
        if rank == 0:
            print("param_dict size", param_dict[('weight',)][0].size)
        xor_results, speed = self._calculate_xor(param_dict)
        if rank == 0:
            print("xor_results size", xor_results[('weight',)].size)
        # xor_results: dict
        # key: tuple with only one element of type string, examples: "weight", "bias"
        # value: tensor, representing the xor result of the params of different layers
        return xor_results, speed

    def save_parity_to_ckpt(self, parity_dict, layer_id_list, ckpt_path='parity_checkpoint.pth'):
        specific_parity = {f'rank_{layer_id}': {} for layer_id in layer_id_list}
        self._extract_specific_parity(parity_dict, specific_parity, layer_id_list)
        # specifi_parity: dict
        # key: string in the form of f'rank_{layer_id}'
        # value: dict
            # key: tuple with only one element of type string (weight, bias)
            # value: tensor, representing the xor result of the params of different layers
        torch.save(specific_parity, os.path.join(ckpt_path, f'parity_rank_{"_".join(map(str, layer_id_list))}.pth'))

    def _extract_parameters(self, model, layer_ids):
        param_dict = {}
        for name, param in model.named_parameters():
            layer_id, param_type = self._parse_layer_and_type(name)
            if layer_id in layer_ids and param_type in self.param_types:
                # layer_id in layer_ids means only layers recorded in the json file will be added to param_dict
                key = (param_type,)
                # param_dict.setdefault(key, []).append(np.frombuffer(param.detach().cpu().numpy().tobytes(), dtype=np.int32))
                param_np = param.detach().cpu().numpy()
                param_bytes = np.frombuffer(param_np.tobytes(), dtype=np.int32)
                param_bytes.shape = param_np.shape
                param_dict.setdefault(key, []).append(param_bytes)
                # param is a tensor, so the value of param_dict is a list of numpy array, 
                # the key of param_dict is a tuple with only one element of type string (weight, bias)
        return param_dict

    def _calculate_xor(self, param_dict):
        # param_dict: dict
        # key: tuple with only one element of type str, example - "weight"
        # value: list of numpy arrays, element - parameter of a layer
        xor_results = {}
        start_time = time.time()
        total_bytes = sum(a.nbytes for param_list in param_dict.values() for a in param_list)
        for key, param_list in param_dict.items():
            xor_result = self.xor_bytes(param_list)
            xor_array = xor_result.view(np.float32)
            xor_tensor = torch.from_numpy(xor_array)
            xor_results[key] = xor_tensor
        elapsed_time = time.time() - start_time
        speed = total_bytes / elapsed_time / (1024 ** 3)
        # xor_results: dict
        # key: tuple with only one element of type str, example - "weight"
        # value: tensor, representing the xor result of the params of different layers
        return xor_results, speed

    def _extract_specific_parity(self, parity_dict, specific_parity, layer_id_list):
        # parity_dict: dict
        # key: tuple with only one element of type string, examples: "weight", "bias"
        # value: tensor, representing the xor result of the params of different layers
        ###
        # specific_parity: dict
        # key: string in the form of f'rank_{layer_id}'
        # value: {}
        ###
        # layer_id_list: list of ints
        for layer_id in layer_id_list:
            for param_type in self.param_types:
                key = (param_type,)
                if key in parity_dict:
                    specific_parity[f'rank_{layer_id}'][key] = parity_dict[key]

    @staticmethod
    def _parse_layer_and_type(name):
        parts = name.split('.')
        layer = parts[0]
        if 'hidden_layers' in layer:
            layer_id = int(parts[1])
        elif layer == 'input_layer':
            layer_id = -1
        elif layer == 'output_layer':
            layer_id = 1000
        param_type = 'weight' if 'weight' in parts[-1] else 'bias'
        return layer_id, param_type

        

# model = LinearModel()
# param_types = ['weight', 'bias']
# with open("../src/models/ddp_config_12_extra.json", "r") as f:
#     redundant_param_keys = json.load(f)

# bxor = ByteXOR(param_types, redundant_param_keys)
# for rank in redundant_param_keys.keys():
#     parity_dict, speed = bxor.process_model(model, rank)
#     print(f'Rank {rank} - Parity calculation speed: {speed:.6f} GB/s')
#     bxor.save_parity_to_ckpt(parity_dict, [rank], ckpt_path='../benchmarks/checkpoints')
    
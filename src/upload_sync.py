import torch
import torch.nn as nn
import socket
import numpy as np
import sys
import json
from multiprocessing import shared_memory
import pickle
import time
from datetime import datetime



class Sync_Tensor_SM:
    def __init__(self, partition_name, model, gpu_id, optimizer=None):
        self.partition_name = partition_name
        self.model = model
        self.gpu_id = str(gpu_id)
        self.optimizer = optimizer
        with open("../src/models/ddp_config_20.json", "r") as f:
            self.config = json.load(f)

    def get_params(self):
        para_dict = {}
        params_to_include = self.config[self.gpu_id]
        for n, p in self.model.named_parameters():
            if n in params_to_include:
                para_dict[n] = p.cpu().detach().numpy()
        if self.optimizer:
            for i, group in enumerate(self.optimizer.param_groups):
                for j, p in enumerate(group['params']):
                    para_dict[f'opt_{i}_{j}'] = p.cpu().detach().numpy()
        return para_dict

    def create_sm(self, data):
        byte_data = pickle.dumps(data)
        total_size = len(byte_data)
        try:
            sm = shared_memory.SharedMemory(create=True, size=total_size, name=self.partition_name)
            buffer = sm.buf
            buffer[:total_size] = byte_data
            print(f"Successfully created shared memory with name: {self.partition_name}") 
            return sm
        except Exception as e:
            print(f"Failed to create shared memory: {e}") 
            return None


    def feed_sm(self, sm):
        params_by_gpu = self.get_params()
        byte_data = pickle.dumps(params_by_gpu)
        buffer = sm.buf
        buffer[:len(byte_data)] = byte_data

    def delete_sm(self, sm):
        sm.unlink()

    def send_req(self, client, state):
        client.send(state.encode('utf-8'))

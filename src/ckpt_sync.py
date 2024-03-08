import sys
import os
import json
import time
import pickle
import torch
from datetime import datetime

def save_checkpoint(model, optimizer, rank, epoch, file_path="/tmp", file_name=None, timestamp_format='%H:%M:%S', to_cpu=True):
    start_time = time.time()
    timestamp = datetime.now().strftime(timestamp_format)

    # Save full model and optimizer states
    full_model_state_dict = model.module.state_dict()
    full_optimizer_state_dict = optimizer.state_dict()

    if to_cpu:  
        full_model_state_dict = {k: v.cpu() for k, v in full_model_state_dict.items()}
    
    if file_name is None:
        file_name = f"checkpoint_rank_{rank}_epoch_{epoch}.pth"

    ckpt_file = os.path.join(file_path, file_name)

    with open(ckpt_file, 'wb') as f:
        pickle.dump({'model': full_model_state_dict, 'optimizer': full_optimizer_state_dict}, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    ckpt_size = os.path.getsize(ckpt_file) / (1024 * 1024)  # MB
    speed = ckpt_size / elapsed_time

    print(f"[{timestamp}] Rank {rank} saved checkpoint of size {ckpt_size:.2f} MB at {speed:.2f} MB/s in {elapsed_time:.2f} seconds.")


def save_sharded_checkpoint(model, optimizer, rank, world_size, epoch, config, partition=None):
    start_time = time.time()
    timestamp = datetime.now().strftime('%H:%M:%S')

    # Get full model and optimizer state
    if partition is None:
        full_model_state_dict = model.module.state_dict()
        full_optimizer_state_dict = optimizer.state_dict()
    else:
        full_model_state_dict = model.module.partitions[partition].state_dict()

    # Shard model state
    shard_model_state_dict_cpu = {}
    for layer_name in config[str(rank)]:
        shard_model_state_dict_cpu[layer_name] = full_model_state_dict[layer_name].cpu()

    # Shard optimizer state
    shard_optimizer_state_dict = {'state': {}, 'param_groups': full_optimizer_state_dict['param_groups']}
    for group in full_optimizer_state_dict['param_groups']:
        for p in group['params']:
            if p in full_optimizer_state_dict['state']:
                shard_optimizer_state_dict['state'][p] = full_optimizer_state_dict['state'][p]

    # Save both model and optimizer states
    ckpt_file = f"/tmp/sharded_checkpoint_rank_{rank}_epoch_{epoch}.pth"
    with open(ckpt_file, 'wb') as f:
        pickle.dump({'model': shard_model_state_dict_cpu, 'optimizer': shard_optimizer_state_dict}, f)

    end_time = time.time()
    elapsed_time = end_time - start_time
    ckpt_size = os.path.getsize(ckpt_file) / (1024 * 1024)  # MB
    speed = ckpt_size / elapsed_time
    print(f"[{timestamp}] Rank {rank} saved checkpoint of size {ckpt_size:.2f} MB at {speed:.2f} MB/s in {elapsed_time:.2f} seconds.")
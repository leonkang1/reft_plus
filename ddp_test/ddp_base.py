import sys
import json
import os
import argparse
import socket
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm

sys.path.append('../src')
sys.path.append('../src/models')
from ddp_model import LinearModel
from ckpt_async import AsyncShardedCheckpoint


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size)
    # create model and move it to GPU with id rank
    model = LinearModel().to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    logging.info((f"Starting training on rank {dist.get_rank()} on node {ddp_model.device}"))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = RandomDataset(args.input_size, args.data_size)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    with open("../src/models/ddp_config_12.json", "r") as f:
        config = json.load(f)
    async_ckpt = AsyncShardedCheckpoint(world_size, config, save_dir="/home/kangxueze/reft2/ddp_test/checkpoints")
    for epoch in tqdm(range(args.epochs)):
        for inputs in dataloader:
            inputs = inputs.to(rank)

            # Forward pass
            outputs = ddp_model(inputs)
            labels = torch.randn(outputs.size()).to(rank)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()
            
        if epoch == 0:
            async_ckpt.save_checkpoint(model, optimizer, rank, epoch, True)

    cleanup()

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Master address: %s", os.environ.get('MASTER_ADDR'))
    logging.info("Master port: %s", os.environ.get('MASTER_PORT'))
    logging.info("cuda visible: %s", os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--data_size', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    args = parser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logging.info("rank %d", rank)
    logging.info("world_size %d", world_size)

    train(rank, world_size, args)

if __name__ == '__main__':
    main()
    
    
    
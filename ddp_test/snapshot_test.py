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
from output import *
from torch.profiler import profile, record_function, ProfilerActivity


sys.path.append('../src')
sys.path.append('../src/models')
from ddp_model import LinearModel
from ckpt_async import AsyncShardedCheckpoint
from ec_coding import ByteXOR
from datetime import datetime



script_dir = os.path.dirname(os.path.abspath(__file__))


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
    # print the number of iterations of the dataloader
    rank0Print(rank, f"number of iterations: {len(dataloader)}", YELLOW)
    with open(os.path.join(script_dir, "../src/models/ddp_config_12.json"), "r") as f:
        config = json.load(f)
    async_ckpt = AsyncShardedCheckpoint(world_size, config, save_dir=os.path.join(script_dir, "checkpoints"))
    for epoch in tqdm(range(args.epochs), desc=f'Rank {rank} Epochs'):
        # rank0Print(rank, f"epoch {epoch}", YELLOW)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True) as prof:
            with record_function("model_training"):

                for i, inputs in enumerate(tqdm(dataloader, desc='Iterations', leave=False)):
                    # rank0Print(rank, f"step {i}", CYAN)
                    inputs = inputs.to(rank)

                    # Forward pass
                    outputs = ddp_model(inputs)
                    labels = torch.randn(outputs.size()).to(rank)
                    loss = criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Optimization step
                    optimizer.step()
                    
                    if args.use_snapshot:
                        async_ckpt.make_snapshot(model, optimizer, rank, epoch, True)
                    
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Get current date and time
    now = datetime.now()
    # Format as a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if args.use_snapshot:
        prof_file_path = os.path.join(script_dir, "trace", f"{timestamp}_{rank}_trace_snapshot.json")
    else:
        prof_file_path = os.path.join(script_dir, "trace", f"{timestamp}_{rank}_trace_no_snapshot.json")
    prof.export_chrome_trace(prof_file_path)
    cleanup()

def main():
    logging.basicConfig(level=logging.INFO)
    # logging.info("Master address: %s", os.environ.get('MASTER_ADDR'))
    # logging.info("Master port: %s", os.environ.get('MASTER_PORT'))
    # logging.info("cuda visible: %s", os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--data_size', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--use_snapshot', action='store_true')
    args = parser.parse_args()

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logging.info("rank %d", rank)
    logging.info("world_size %d", world_size)

    train(rank, world_size, args)

if __name__ == '__main__':
    main()
    

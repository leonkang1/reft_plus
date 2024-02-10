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
from ckpt_async import AsyncCheckpoint
from datetime import datetime



script_dir = os.path.dirname(os.path.abspath(__file__))


def non_blocking_make_snapshot(model, optimizer, use_copy_):
    if use_copy_:
        for param_name, model_tensor_gpu in model.state_dict().items():
            model_tensor_cpu = torch.empty_like(model_tensor_gpu, device='cpu')
            model_tensor_cpu.copy_(model_tensor_gpu, non_blocking=True)
        # torch.cuda.synchronize()
    else:
        cpu_state_dict = {key: value.cpu() for key, value in model.state_dict().items()}

    optimizer_state_dict = optimizer.state_dict()
    # traverse all the values in shard_optimizer_state_dict['state'] and convert them to cpu if they are tensors
    for k, v in optimizer_state_dict['state'].items():
        if torch.is_tensor(v):
            if use_copy_: 
                optimizer_tensor_gpu = v
                optimizer_tensor_cpu = torch.empty_like(optimizer_tensor_gpu, device='cpu')
                optimizer_tensor_cpu.copy_(optimizer_tensor_gpu, non_blocking=True) 
            else:
                optimizer_tensor_cpu = v.cpu()

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def train(args):
    device = 'cuda:0'
    # create model and move it to GPU with id rank
    model = LinearModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    dataset = RandomDataset(args.input_size, args.data_size)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=40)
    # print the number of iterations of the dataloader

    async_ckpt = AsyncCheckpoint(save_dir=os.path.join(script_dir, "checkpoints"))
    
    for epoch in tqdm(range(args.epochs)):
        # rank0Print(rank, f"epoch {epoch}", YELLOW)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True, with_stack=True) as prof:
            with record_function("model_training"):
                for i, inputs in enumerate(tqdm(dataloader, desc='Iterations', leave=False)):
                    # rank0Print(rank, f"step {i}", CYAN)
                    inputs = inputs.to(device)
                    optimizer.zero_grad()
                    # Forward pass
                    outputs = model(inputs)
                    labels = torch.randn(outputs.size(), device=device)
                    loss = criterion(outputs, labels)
                    # Backward pass
                    loss.backward()
                    # Optimization step
                    optimizer.step()
                    if args.use_snapshot and args.async_snapshot:
                        async_ckpt.make_snapshot(model, optimizer, epoch, use_copy_=args.use_copy, is_partition=True)
                    if args.use_snapshot and args.non_blocking_copy:
                        non_blocking_make_snapshot(model, optimizer, args.use_copy)
                    prof.step()
                    
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # Get current date and time
    now = datetime.now()
    # Format as a string
    timestamp = now.strftime("%m%d_%H%M%S")
    prof_file_path = os.path.join(script_dir, "trace", f"{timestamp}-use_copy_{args.use_copy}-use_snapshot_{args.use_snapshot}-trace.json")
    prof.export_chrome_trace(prof_file_path)

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
    parser.add_argument('--use_copy', action='store_true')
    parser.add_argument('--non_blocking_copy', action='store_true')
    parser.add_argument('--async_snapshot', action='store_true')
    args = parser.parse_args()
 
    train(args)

if __name__ == '__main__':
    main()
    

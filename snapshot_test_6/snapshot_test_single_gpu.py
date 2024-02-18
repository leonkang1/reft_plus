import sys
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import contextmanager
import time
import datetime

sys.path.append('..')
sys.path.append('../src')
sys.path.append('../src/models')
from output import *
from ddp_model import LinearModel
from async_snapshot import AsyncCheckpoint




script_dir = os.path.dirname(os.path.abspath(__file__))


def non_async_make_snapshot(model, optimizer, use_copy_, use_timer, step_cnt, timer_record_file, non_blocking_copy, snapshot_stream, device):
    if use_timer and step_cnt > 10:
        start_time = time.perf_counter()
    snapshot_stream.wait_stream(torch.cuda.default_stream(device))
    with torch.cuda.stream(snapshot_stream):
        if use_copy_:
            for param_name, model_tensor_gpu in model.state_dict().items():
                model_tensor_cpu = torch.empty_like(model_tensor_gpu, device='cpu')
                model_tensor_cpu.copy_(model_tensor_gpu, non_blocking=non_blocking_copy) #pin_memory
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
                    optimizer_tensor_cpu.copy_(optimizer_tensor_gpu, non_blocking=non_blocking_copy) 
                else:
                    optimizer_tensor_cpu = v.cpu()
                    
        if use_timer and step_cnt > 10:
            end_time = time.perf_counter()
            timer_record_file.write(f"step: {step_cnt}\n")
            timer_record_file.write(f"snapshot time: {end_time - start_time}\n")

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def train(args):
    @contextmanager
    def dummy_context():
        print("use dummy context")
        yield None

    if args.enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
            record_shapes=True, 
            with_stack=True,
        )
    else:
        profiler_context = dummy_context()

    
    device = 'cuda:0'
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # create model and move it to GPU with id rank
    model = LinearModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    dataset = RandomDataset(args.input_size, args.data_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=40)
    # print the number of iterations of the dataloader

    async_ckpt = AsyncCheckpoint(save_dir=os.path.join(script_dir, "checkpoints"))
    
    snapshot_stream = torch.cuda.Stream()
    
    for epoch in tqdm(range(args.epochs)):
        # rank0Print(rank, f"epoch {epoch}", YELLOW)
        step_cnt = 0
        if args.use_timer:
            time_stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
            timer_record_file_path = os.path.join(script_dir, "timer_record", f"{time_stamp}_option_{args.option_num}_timer_record.txt")
            timer_record_file = open(timer_record_file_path, "w")
            timer_record_file.write(str(args) + '\n')
            snapshot_timer_record_file_path = os.path.join(script_dir, "timer_record", f"{time_stamp}_option_{args.option_num}_snapshot_timer_record.txt")
            snapshot_timer_record_file = open(snapshot_timer_record_file_path, "w")
            train_start_time = time.perf_counter()
        if args.use_snapshot and args.async_snapshot:
            async_checkpoint_thread_list = []
        with profiler_context as prof:
            for i, inputs in enumerate(tqdm(dataloader, desc='Iterations', leave=False)):
                with record_function(f"model_training_{step_cnt}"):
                    if args.use_timer and step_cnt > 10:
                        timer_record_file.write(f"step: {step_cnt}\n")
                        step_start_time = time.perf_counter()
                        start_time = time.perf_counter()
                    inputs = inputs.to(device)
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"input transfer time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    optimizer.zero_grad()
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"zero_grad time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    # Forward pass
                    outputs = model(inputs)
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"forward pass time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    labels = torch.randn(outputs.size(), device=device)
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"label generation time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    loss = criterion(outputs, labels)
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"loss calculation time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    # Backward pass
                    loss.backward()
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"backward pass time: {end_time - start_time}\n")
                        start_time = time.perf_counter()
                    # Optimization step
                    optimizer.step()
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"optimization step time: {end_time - start_time}\n")
                        timer_record_file.write(f"step time: {end_time - step_start_time}\n")
                        start_time = time.perf_counter()
                    if args.use_snapshot:
                        if not args.use_timer:
                            snapshot_timer_record_file = None
                        if args.async_snapshot:
                            checkpoint_thread = async_ckpt.make_snapshot(model, optimizer, epoch, use_timer=args.use_timer, step_cnt=step_cnt, timer_record_file=snapshot_timer_record_file, use_copy_=args.use_copy, snapshot_stream=snapshot_stream, device=device, non_blocking_copy=args.non_blocking_copy, use_pin_memory=args.use_pin_memory)
                            async_checkpoint_thread_list.append(checkpoint_thread)
                        else:
                            non_async_make_snapshot(model, optimizer, args.use_copy, args.use_timer, step_cnt, snapshot_timer_record_file, args.non_blocking_copy, snapshot_stream, device)
                        
                    if args.use_timer and step_cnt > 10:
                        end_time = time.perf_counter()
                        timer_record_file.write(f"invoke snapshot time: {end_time - start_time}\n")
                    if args.enable_profiling:
                        prof.step()
                    step_cnt += 1
                    
    torch.cuda.synchronize()
    
    if args.enable_profiling:
        time_stamp = datetime.datetime.now().strftime("%m%d-%H%M%S")
        trace_arg_file_path = os.path.join(script_dir, "trace", f"{time_stamp}_option_{args.option_num}_trace_arg.txt")
        trace_arg_file = open(trace_arg_file_path, "w")
        trace_arg_file.write(str(args) + '\n')            
        prof.export_chrome_trace(os.path.join(script_dir, "trace", f"{time_stamp}_option_{args.option_num}_trace.json"))
                    
    if args.use_timer:
        train_end_time = time.perf_counter()
        timer_record_file.write(f"train time: {train_end_time - train_start_time}\n")
        timer_record_file.close()
        if args.use_snapshot and args.async_snapshot:
            for thread in async_checkpoint_thread_list:
                thread.join()
        snapshot_timer_record_file.close()


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
    parser.add_argument("--enable_profiling", action="store_true",
                    help="Enable profiling of the training loop")
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--use_timer', action='store_true')
    parser.add_argument('--use_synthetic_input', action='store_true')
    parser.add_argument('--option_num', type=int)
    parser.add_argument('--use_pin_memory', action='store_true')
    args = parser.parse_args()
    
    nprint(args, CYAN)
 
    train(args)

if __name__ == '__main__':
    main()
    

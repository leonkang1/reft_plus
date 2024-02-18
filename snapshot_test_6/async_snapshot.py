from concurrent.futures import ThreadPoolExecutor
import asyncio
import torch
import aiofiles
import threading
import pickle
import io
import os
from datetime import datetime
import time

class AsyncCheckpoint:
    def __init__(self, save_dir, buffer_size=4096, max_workers=1):
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.allreduce_semaphore = asyncio.Semaphore(1)
        self.thread_lock = threading.Lock()

    def save_checkpoint(self, model, optimizer, file_name):
        start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] snapshot started...")
        
        asyncio.run(self.allreduce_semaphore.acquire())
        
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_thread,
            args=(model, optimizer, file_name, start_time)
        )
        checkpoint_thread.start()

    def _checkpoint_thread(self, model, optimizer, file_name, start_time):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._save_checkpoint(model, optimizer, file_name))
        finally:
            loop.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        file_path = os.path.join(self.save_dir, file_name)
        ckpt_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        speed = ckpt_size / elapsed_time
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Checkpoint size {ckpt_size:.2f} MB, speed {speed:.2f} MB/s, elapsed {elapsed_time:.2f} sec.")

    async def _save_checkpoint(self, model, optimizer, file_name):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        buffer = io.BytesIO()
        current_loop = asyncio.get_event_loop()
        self.allreduce_semaphore.release()
        
        await current_loop.run_in_executor(self.executor, torch.save, state, buffer)
        buffer.seek(0)
        file_path = os.path.join(self.save_dir, file_name)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(buffer.read())
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] saved")
        
    def make_snapshot(self, model, optimizer, epoch, use_timer, step_cnt, timer_record_file, use_copy_, snapshot_stream, device, non_blocking_copy, use_pin_memory):
        # with self.thread_lock:
        #     asyncio.run(self.allreduce_semaphore.acquire())

        checkpoint_thread = threading.Thread(
            target=self._snapshot_thread,
            args=(model, optimizer, epoch, use_copy_, use_timer, step_cnt, timer_record_file, snapshot_stream, device, non_blocking_copy, use_pin_memory)
        )
        checkpoint_thread.start()
        return checkpoint_thread
        
    def _snapshot_thread(self, model, optimizer, epoch, use_copy_, use_timer, step_cnt, timer_record_file, snapshot_stream, device, non_blocking_copy, use_pin_memory):
        if use_timer and step_cnt > 10:
            start_time = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_snapshot(model, optimizer, epoch, use_copy_, snapshot_stream, device, non_blocking_copy, use_pin_memory))
        finally:
            loop.close()
        if use_timer and step_cnt > 10:
            end_time = time.perf_counter()
            timer_record_file.write(f"step: {step_cnt}\n")
            timer_record_file.write(f"snapshot time: {end_time - start_time}\n")

    async def _make_snapshot(self, model, optimizer, epoch, use_copy_, snapshot_stream, device, non_blocking_copy, use_pin_memory):
        snapshot_stream.wait_stream(torch.cuda.default_stream(device))
        with torch.cuda.stream(snapshot_stream):
            if use_copy_:
                for param_name, model_tensor_gpu in model.state_dict().items():
                    model_tensor_cpu = torch.empty_like(model_tensor_gpu, device='cpu')
                    if use_pin_memory:
                        model_tensor_cpu = model_tensor_cpu.pin_memory()
                    model_tensor_cpu.copy_(model_tensor_gpu, non_blocking=non_blocking_copy)
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
                        if use_pin_memory:
                            optimizer_tensor_cpu = optimizer_tensor_cpu.pin_memory()
                        optimizer_tensor_cpu.copy_(optimizer_tensor_gpu, non_blocking=non_blocking_copy) 
                    else:
                        optimizer_tensor_cpu = v.cpu()
        # torch.cuda.synchronize()
        # self.allreduce_semaphore.release()


class AsyncShardedCheckpoint:
    def __init__(self, world_size, config, save_dir, buffer_size=4096, max_workers=4):
        self.world_size = world_size
        self.config = config
        self.save_dir = save_dir
        self.buffer_size = buffer_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.allreduce_semaphore = asyncio.Semaphore(1)
        self.thread_lock = threading.Lock()

    def save_checkpoint(self, model, optimizer, rank, epoch, is_partition=False):
        start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Rank {rank} snapshot started...")
        with self.thread_lock:
            asyncio.run(self.allreduce_semaphore.acquire())
        
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_thread,
            args=(model, optimizer, rank, epoch, is_partition, start_time)
        )
        checkpoint_thread.start()

    def _checkpoint_thread(self, model, optimizer, rank, epoch, is_partition, start_time):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._save_sharded_checkpoint(model, optimizer, rank, epoch, is_partition))
        finally:
            loop.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        ckpt_file = os.path.join(self.save_dir, f"sharded_checkpoint_rank_{rank}_epoch_{epoch}.pth")
        ckpt_size = os.path.getsize(ckpt_file) / (1024 * 1024)
        speed = ckpt_size / elapsed_time
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Rank {rank} Checkpoint size {ckpt_size:.2f} MB, speed {speed:.2f} MB/s, elapsed {elapsed_time:.2f} sec.")

    async def _save_sharded_checkpoint(self, model, optimizer, rank, epoch, is_partition):
        if is_partition: # I think that the meaning here is that the model is doing MP
            shard_model_state_dict_cpu = {k: model.state_dict()[k].cpu() for k in self.config[str(rank)]}
        else: # It is doing DP, so I need to use the module to get the state_dict
            full_model_state_dict = model.module.state_dict()
            shard_model_state_dict_cpu = {k: full_model_state_dict[k].cpu() for k in self.config[str(rank)]}
        
        full_optimizer_state_dict = optimizer.state_dict()
        shard_optimizer_state_dict = {
            'state': {p: full_optimizer_state_dict['state'][p] for group in full_optimizer_state_dict['param_groups'] for p in group['params'] if p in full_optimizer_state_dict['state']},
            'param_groups': full_optimizer_state_dict['param_groups']
        }
        
        buffer = io.BytesIO()
        current_loop = asyncio.get_event_loop()
        self.allreduce_semaphore.release()
        await current_loop.run_in_executor(self.executor, pickle.dump, {'model': shard_model_state_dict_cpu, 'optimizer': shard_optimizer_state_dict}, buffer)
        buffer.seek(0)
        ckpt_file = os.path.join(self.save_dir, f"sharded_checkpoint_rank_{rank}_epoch_{epoch}.pth")
        async with aiofiles.open(ckpt_file, "wb") as f:
            await f.write(buffer.read())
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Rank {rank} saved.")
        
    def make_snapshot(self, model, optimizer, rank, epoch, use_copy_=False, is_partition=False):
        start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Rank {rank} snapshot started...")
        with self.thread_lock:
            asyncio.run(self.allreduce_semaphore.acquire())
        
        checkpoint_thread = threading.Thread(
            target=self._snapshot_thread,
            args=(model, optimizer, rank, epoch, is_partition, start_time, use_copy_)
        )
        checkpoint_thread.start()
        
    def _snapshot_thread(self, model, optimizer, rank, epoch, is_partition, start_time, use_copy_):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_sharded_snapshot(model, optimizer, rank, epoch, is_partition, use_copy_))
        finally:
            loop.close()

    async def _make_sharded_snapshot(self, model, optimizer, rank, epoch, is_partition, use_copy_):
        if is_partition:
            if use_copy_:
                shard_model_state_dict_cpu = {}
                for k in self.config[str(rank)]:
                    model_tensor_gpu = model.state_dict()[k]
                    model_tensor_cpu = torch.empty_like(model_tensor_gpu, device='cpu')
                    model_tensor_cpu.copy_(model_tensor_gpu)
                    # torch.cuda.synchronize()
                    shard_model_state_dict_cpu[k] = model_tensor_cpu
            else:
                shard_model_state_dict_cpu = {k: model.state_dict()[k].cpu() for k in self.config[str(rank)]}
        else:
            if use_copy_:
                full_model_state_dict = model.module.state_dict()
                shard_model_state_dict_cpu = {}
                for k in self.config[str(rank)]:
                    model_tensor_gpu = full_model_state_dict[k]
                    model_tensor_cpu = torch.empty_like(model_tensor_gpu, device='cpu')
                    model_tensor_cpu.copy_(model_tensor_gpu)
                    # torch.cuda.synchronize()
                    shard_model_state_dict_cpu[k] = model_tensor_cpu
            else:
                full_model_state_dict = model.module.state_dict()
                shard_model_state_dict_cpu = {k: full_model_state_dict[k].cpu() for k in self.config[str(rank)]}
        
        full_optimizer_state_dict = optimizer.state_dict()
        shard_optimizer_state_dict = {
            'state': {p: full_optimizer_state_dict['state'][p] for group in full_optimizer_state_dict['param_groups'] for p in group['params'] if p in full_optimizer_state_dict['state'] },
            'param_groups': full_optimizer_state_dict['param_groups']
        }
        # traverse all the values in shard_optimizer_state_dict['state'] and convert them to cpu if they are tensors
        for k, v in shard_optimizer_state_dict['state'].items():
            if torch.is_tensor(v):
                if use_copy_: 
                    optimizer_tensor_gpu = full_optimizer_state_dict['state'][k]
                    optimizer_tensor_cpu = torch.empty_like(optimizer_tensor_gpu, device='cpu')
                    optimizer_tensor_cpu.copy_(optimizer_tensor_gpu) 
                    # torch.cuda.synchronize()
                    shard_optimizer_state_dict['state'][k] = optimizer_tensor_cpu
                else:
                    shard_optimizer_state_dict['state'][k] = v.cpu()
        # torch.cuda.synchronize()
        self.allreduce_semaphore.release()
        print(f"rank {rank} finish snapshotting")
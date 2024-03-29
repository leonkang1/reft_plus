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
from ec_coding import ByteXOR


class ARSckpt:
    def __init__(self, world_size, config, save_dir, buffer_size=4096, max_workers=4,redundant_level=0):
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
        if is_partition:
            shard_model_state_dict_cpu = {k: model.state_dict()[k].cpu() for k in self.config[str(rank)]}
        else:
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
            await f.flush() 
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Rank {rank} saved.")

from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
from multiprocessing import shared_memory
import json
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
import threading


import json
import pickle
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from multiprocessing import shared_memory
import time  

class Async_Tensor_SM:
    def __init__(self, partition_name, model, gpu_id, optimizer=None):
        self.partition_name = partition_name
        self.model = model
        self.gpu_id = str(gpu_id)
        self.optimizer = optimizer
        with open("../src/models/ddp_config_20.json", "r") as f:
            self.config = json.load(f)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.async_lock = asyncio.Lock()

    def get_sharded_params(self):
        para_dict = {}
        total_size = 0
        params_to_include = ['module.' + x for x in self.config[self.gpu_id]]
        # print(f"Params to include: {params_to_include}") 
        for n, p in self.model.named_parameters():
            # print(f"Model param: {n}")  # Debug print
            if n in params_to_include:
                # print(f"Parameter name: {n}")
                # print(f"Parameter tensor: {p}")
                # print(f"Parameter size: {p.size()}")
                # print(f"Parameter numel: {p.numel()}")
                para_dict[n] = p.cpu().detach().numpy()
                total_size += p.numel() * 4  
        print(f"Total size calculated: {total_size}")  
        return para_dict, total_size

    def create_sm(self):
        sm_thread = threading.Thread(target=self._create_sm_thread)
        sm_thread.start()

    def _create_sm_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._create_sm_async())
        finally:
            loop.close()

    async def _create_sm_async(self):
        timestamp_start = datetime.now().strftime('%H:%M:%S.%f')
        print(f"[{timestamp_start}] Started creating shared memory.")
        
        byte_data = pickle.dumps(self.get_sharded_params())
        total_size = len(byte_data) 
        print(total_size)
        model_size_MB = total_size / (1024 * 1024) 

        print(f"Model size: {model_size_MB:.2f} MB")

        loop = asyncio.get_event_loop()
        start_time = time.time() 
        
        async with self.async_lock:
            try:
                await loop.run_in_executor(self.executor, self._create_sm, byte_data, total_size)
                timestamp_end = datetime.now().strftime('%H:%M:%S.%f')
                print(f"[{timestamp_end}] Successfully created shared memory with name: {self.partition_name}") 
            except Exception as e:
                timestamp_end = datetime.now().strftime('%H:%M:%S.%f')
                print(f"[{timestamp_end}] Failed to create shared memory: {e}") 

        end_time = time.time()  
        speed_MB_s = model_size_MB / (end_time - start_time)  
        print(f"Transfer speed to SM: {speed_MB_s:.2f} MB/s")

    def _create_sm(self, byte_data, total_size):
        sm = shared_memory.SharedMemory(create=True, size=total_size, name=self.partition_name)
        buffer = sm.buf
        # buffer[:total_size] = byte_data

    def feed_sm(self, sm):
        feed_thread = threading.Thread(target=self._feed_sm_thread, args=(sm,))
        feed_thread.start()

    def _feed_sm_thread(self, sm):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._feed_sm_async(sm))
        finally:
            loop.close()

    async def _feed_sm_async(self, sm):
        timestamp_start = datetime.now().strftime('%H:%M:%S.%f')
        print(f"[{timestamp_start}] Started feeding data to shared memory.")
        
        params_by_gpu = self.get_sharded_params()
        byte_data = pickle.dumps(params_by_gpu)
        loop = asyncio.get_event_loop()
        
        async with self.async_lock:
            await loop.run_in_executor(self.executor, self._feed_sm_buffer, sm, byte_data)
        
        timestamp_end = datetime.now().strftime('%H:%M:%S.%f')
        print(f"[{timestamp_end}] Completed feeding data to shared memory.")

    def _feed_sm_buffer(self, sm, byte_data):
        buffer = sm.buf
        buffer[:len(byte_data)] = byte_data

    def delete_sm(self, sm):
        sm.unlink()

    def send_req(self, client, state):
        client.send(state.encode('utf-8'))

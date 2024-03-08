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
import pickle
import json
import numpy as np
import sys
import time

sys.path.append('../src')
sys.path.append('../src/models')
from ec_coding import ByteXOR

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input_layer = nn.Linear(512, 4096)
        self.hidden_layers = nn.ModuleList([nn.Linear(4096, 4096) for _ in range(12)])
        self.output_layer = nn.Linear(4096, 512)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


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

def train(model, rank, world_size, args):
    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    ddp_model = DistributedDataParallel(model, device_ids=[device_id])
    logging.info((f"Starting training on rank {dist.get_rank()} on node {ddp_model.device}"))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    dataset = RandomDataset(args.input_size, args.data_size)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    for epoch in tqdm(range(args.epochs)):
        for inputs in dataloader:
            inputs = inputs.to(device_id)

            # Forward pass
            outputs = ddp_model(inputs)
            labels = torch.randn(outputs.size()).to(device_id)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

    cleanup()

def xor_parameters(tensor1, tensor2):
    byte_xor = ByteXOR(param_types=["weight", "bias"], redundant_param_keys={str(rank): [] for rank in [0, 1, 2, 3]})
    # arrays_list = [np.frombuffer(t.detach().cpu().numpy().tobytes(), dtype=np.int32) for t in [tensor1, tensor2]]
    arrays_list = []
    for t in [tensor1, tensor2]:
        t_np = t.detach().cpu().numpy()
        t_bytes = np.frombuffer(t_np.tobytes(), dtype=np.int32)
        t_bytes.shape = t_np.shape
        arrays_list.append(t_bytes)
    xor_result = byte_xor.xor_bytes(arrays_list)
    xor_array = xor_result.view(np.float32)
    return torch.from_numpy(xor_array)


def recover_shards(failed_node, shard_recv_tensor_dict, parity_recv_tensor_dict):
    """
    failed_node: int
    the rank of the failed node
    
    shard_recv_tensor_list: list 2D
    the first index is the rank of a healthy node, the second index represents the tensor's meaning,
    which is consistent of what is recorded in ddp_config_12.json
    The length of the list should be 4 * 3 * 2
    
    parity_recv_tensor_list: list 2D
    The first index is the rank of a healthy node, the second index represents the tensor's meaning
    0 represents the weight parity, 1 represents the bias parity
    """
    layer_info_dir = "config"
    shard_layer_path = os.path.join(layer_info_dir, "ddp_config_12.json")
    parity_layer_path = os.path.join(layer_info_dir, "ddp_config_12_extra.json")
    
    with open(shard_layer_path, "r") as file:
        shard_name_dict = json.load(file)
    with open(parity_layer_path, "r") as file:
        parity_name_dict = json.load(file)
        
    recovered_shards = {}
        
    missing_layers = shard_name_dict[str(failed_node)]
        
    for missing_layer in missing_layers:
        for parity_rank, parity_layers in parity_name_dict.items():
            if missing_layer in parity_layers:
                missing_parity_type = missing_layer.split('.')[2]
                missing_layer_num = missing_layer.split('.')[1]
                recovered_value = parity_recv_tensor_dict[int(parity_rank)][missing_parity_type]

                for layer in parity_layers:
                    layer_num = layer.split('.')[1]
                    parity_type = layer.split('.')[2]
                    if layer_num != missing_layer_num and parity_type == missing_parity_type:
                        recovered_value = xor_parameters(recovered_value, shard_recv_tensor_dict[layer])
                
                recovered_shards[missing_layer] = recovered_value    
    """
    return recovered_shards: dict
    key: str, missing layer name of the shards of the failed node
    value: the corresponding tensor of the recovered missing layers
    note: the sequence of the keys is identical to that in shard_layer_dict
    """
    
    return recovered_shards


def shard_check(current_rank, rank_list, model_parameters):
    ckpt_dir = "checkpoints"
    for rank in rank_list:
        shard_path = os.path.join(ckpt_dir, f"sharded_checkpoint_rank_{rank}_epoch_0.pth")
        with open(shard_path, "rb") as file:
            current_rank_shard_tensor_dict = pickle.load(file)
            current_rank_shard_tensor_dict = current_rank_shard_tensor_dict["model"]
            for shard_name in current_rank_shard_tensor_dict.keys():
                try:
                    assert(torch.equal(model_parameters[shard_name], current_rank_shard_tensor_dict[shard_name].to(current_rank)))
                except:
                    print("shard name", shard_name)
                    print("get", model_parameters[shard_name])
                    print("original", current_rank_shard_tensor_dict[shard_name])
                    raise()
                
def parity_check(current_rank, rank_list, parity_dict):
    ckpt_dir = "checkpoints"
    for rank in rank_list:
        rank_list = [2,3]
        parity_path = os.path.join(ckpt_dir, f"parity_rank_{rank}.pth")
        with open(parity_path, "rb") as file:
            parity = torch.load(file)
            parity = parity[f"rank_{rank}"]
            try:
                assert(torch.equal(parity[("weight",)].to(current_rank), parity_dict[rank]["weight"]))
            except:
                print("rank", rank)
                print("original", parity[("weight",)])
                print("recv", parity_dict[rank]["weight"])
            try:
                assert(torch.equal(parity[("bias",)].to(current_rank), parity_dict[rank]["bias"]))
            except:
                print("rank", rank)
                print("original", parity[("bias",)])
                print("recv", parity_dict[rank]["bias"])
            
        

def recover_node(failed_node, shard_shape_dict, parity_shape_dict):
    """Recover a failed node using snapshots and parity data from other nodes."""
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = current_rank % torch.cuda.device_count()
    
    ckpt_dir = "checkpoints"
    layer_info_dir = "config"
    shard_layer_path = os.path.join(layer_info_dir, "ddp_config_12.json")
    parity_layer_path = os.path.join(layer_info_dir, "ddp_config_12_extra.json")
    with open(shard_layer_path, "r") as file:
        shard_name_dict = json.load(file)
    with open(parity_layer_path, "r") as file:
        parity_layer_dict = json.load(file)
    
    model_parameters = {}
    for shard_name, shard_shape in shard_shape_dict.items():
        model_parameters[shard_name] = torch.zeros(shard_shape, device=device_id)
        
    healthy_rank_list = list(range(world_size))
    healthy_rank_list.remove(failed_node)
    all_gather_group = dist.new_group(healthy_rank_list)

    # Each node sends its snapshot and parity to the failed node
    if current_rank != failed_node:
        # Get the parity and ckpt data
        parity_path = os.path.join(ckpt_dir, f"parity_rank_{current_rank}.pth")
        with open(parity_path, "rb") as file:
            parity = torch.load(file)
            parity = parity[f"rank_{current_rank}"]
            
        shard_path = os.path.join(ckpt_dir, f"sharded_checkpoint_rank_{current_rank}_epoch_0.pth")
        with open(shard_path, "rb") as file:
            current_rank_shard_tensor_dict = pickle.load(file)
            current_rank_shard_tensor_dict = current_rank_shard_tensor_dict["model"]
        # logging.info("load tensors ready")
        # Get the model parameters of the current rank
        for shard_name in current_rank_shard_tensor_dict.keys():
            model_parameters[shard_name] = current_rank_shard_tensor_dict[shard_name].to(device_id)
            
        # send the shard and parity to failed node first, with non-blocking method
        for shard_name in current_rank_shard_tensor_dict.keys():
            req = dist.isend(tensor=model_parameters[shard_name], dst=failed_node)
            req.wait()
            # dist.send(tensor=model_parameters[shard_name], dst=failed_node)
            # logging.info("send shard_name %s", shard_name)
        
        # logging.info("Send shard to failed node")
            # Param sent order is identical to the order that these params are stored which is identical to the order in the json file
        # 把ckpt里的hidden_layer的tensor发送过去        
        # print("current rank", current_rank)
        for parity_type, parity_tensor in parity.items():
            # print(parity_type, parity_tensor)
            req = dist.isend(tensor=parity_tensor.to(device_id), dst=failed_node)
            req.wait()
            # dist.send(tensor=parity_tensor.to(device_id), dst=failed_node)
            # logging.info("send parity %s %s", current_rank, parity_type)
        # 把parity的tensor发送过去
        # logging.info("send parities to failed node")
        
        # send the shards to every healthy node
        for i, shard_name in enumerate(shard_name_dict[str(current_rank)]):
            if "hidden" in shard_name:
                all_gather_shard_list = [torch.zeros_like(model_parameters[shard_name]).to(device_id) for _ in range(world_size - 1)]
                dist.all_gather(all_gather_shard_list, model_parameters[shard_name], group=all_gather_group)
                for j in range(len(healthy_rank_list)):
                    if healthy_rank_list[j] != current_rank:
                        model_parameters[shard_name_dict[str(healthy_rank_list[j])][i]] = all_gather_shard_list[j]
        
        all_gather_other_rank_list = healthy_rank_list.copy()
        all_gather_other_rank_list.remove(current_rank)
        shard_check(current_rank, all_gather_other_rank_list, model_parameters)
        
        for shard_name in shard_name_dict[str(failed_node)]:
            dist.broadcast(model_parameters[shard_name], src=failed_node)
        shard_check(current_rank, [failed_node], model_parameters)
    # Recover the parameters on the failed node
    # ...

    # After recovery, the failed node sends its snapshot to every other node
    else:
        # After I've got the layer information, I can initialized the receiving tensor now
        parity_type_list = ["weight", "bias"]
        parity_dict = {}
        for i in range(world_size):
            parity_dict[i] = {}
            for parity_type in parity_type_list:
                parity_dict[i][parity_type] = torch.zeros(parity_shape_dict[parity_type], device=device_id)
                
        recv_requests = [] 
        
        for rank, rank_shard_name_list in shard_name_dict.items():
            if int(rank) != failed_node:
                for shard_name in rank_shard_name_list:
                    recv_requests.append(dist.irecv(model_parameters[shard_name], src=int(rank)))       
                    # dist.recv(model_parameters[shard_name], src=int(rank))
                    # logging.info("recv shard_name %s", shard_name)
        
        for rank in range(world_size):
            if rank != failed_node:
                for parity_type in parity_type_list:
                    recv_requests.append(dist.irecv(parity_dict[rank][parity_type], src=rank))
                    # dist.recv(parity_dict[rank][parity_type], src=rank)
                    # logging.info("recv parity %s %s", rank, parity_type)
        
        
        for req in recv_requests:
            req.wait()
                    
        healthy_rank_list = list(range(4))
        healthy_rank_list.remove(failed_node)
        
        shard_check(current_rank, healthy_rank_list, model_parameters)
        parity_check(current_rank, healthy_rank_list, parity_dict)
        # Now I'v got all the shards and parities from healthy nodes
        # Next I need to restore the shard of the failed node
        recovered_shard_dict = recover_shards(failed_node, model_parameters, parity_dict)
        for shard_name, shard_tensor in recovered_shard_dict.items():
            model_parameters[shard_name] = shard_tensor.to(device_id)
        shard_check(current_rank, [0], model_parameters)
        
    #     # Next I need to send the restored shard to all the healthy nodes 
        for shard_name in shard_name_dict[str(failed_node)]:
            dist.broadcast(model_parameters[shard_name], src=failed_node)
            
    return model_parameters
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--input_size', default=512, type=int)
    parser.add_argument('--data_size', default=100, type=int)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--failed_node', default=0, type=int,
                        help='The id of failed node')
    args = parser.parse_args()

    
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.cuda.device_count()
    model = LinearModel().to(device_id)
    shard_shape_dict = {name: list(param.size()) for name, param in model.named_parameters()}
    parity_shape_dict = {"weight": [4096, 4096], "bias": [4096]}
    model_parameters = recover_node(failed_node=0, shard_shape_dict=shard_shape_dict, parity_shape_dict=parity_shape_dict)
    model.load_state_dict(model_parameters)

    train(model, rank, world_size, args)

if __name__ == '__main__':
    main()
    
    
    
    
    
"""
Debug notes:

1. Don't specify MASTER_PORT using os.environ in the script. When you are using DDP with single-node multi-GPU situation,
   specifying MASTER_PORT manually would lead the program to be stuck, idk the reason. By default, DDP would choose 29500 as 
   the MASTER_PORT, it doesn't matter if you specify MASTER_PORT in the command line.

2. When you run script in single-node multi-GPU situation, --nnodes and --nproc-per-node can be set as arguments.
   CUDA_VISIBLE_DEVICES can be set as envs. Other envs like rank and world_size are set by torch.distributed.run directly.

3. When you run script in multi-node situation, you should run the command line like this:
   python -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=4 \
    --rdzv_id=$rdzv_id \
    --rdzv_backend=c10d \
    --rdzv_endpoint=HOST_NODE_ADDR \
    YOUR_TRAINING_SCRIPT.py (--arg1 ... --arg2 ...)
   Besides nnodes and nproc-per-node, you also need to specify rdzv_id, rdzv_backend and rdzv_endpoint.
   rdzv_id is the job id, you can specify any int you want. Just make sure every node use the same id.
   rdzv_endpoint should be the address of the master node. All other nodes will use this address to discover each other and 
   set up the distributed training.
   rdzv_backend can simply be set as c10d. Meaning is unstudied.

4. About logging info
   You can't use logging.info("a", "b") just like in print(). If you want to combine multiple strings, you need to write as
   logging.info("a %s", "b")

5. Use parser to get argument in command lines.
   1. parser = argparse.ArgumentParser()
   2. parser.add_argument('--epochs', default=2, type=int, metavar='N',
          help='number of total epochs to run')
      You can specify argument name, default value, argument type, help content
   3. args = parser.parse_args()
   4. args.epochs can get the value.
   
6.  When using dist.isend to send data, it seems that if I collect all the sending request in a list, and make them wait after 
    every request has execute isend, the sent data might influence each other. For example, the sending code I implemented
    before is like:
    
    shard_send_requests = []
    parity_send_requests = []
    for shard in shard_list:
        shard_send_requests.append(dist.isend(shard, src=...))
    for parity in parity_list:
        parity_send_requests.append(dist.isend(parity, src=...))
        
    for req in shard_send_requests:
        req.wait()
    for req in parity_send_requests:
        req.wait()
        
    Some strange problems occur at this implementation. 
    The parity data sent by node 1 is:
    tensor([[-5.6711e-03, -5.5127e-04, 8.0800e-04, ..., 7.8006e-03,
    6.6992e-05, 9.6430e-03],
    [-1.6915e-03, 9.7358e-02, -6.7334e-03, ..., 2.1382e-02,
    -1.1618e-03, -1.0632e-02],
    [-3.0551e-03, -1.0185e-01, 6.1934e-02, ..., 1.1891e-02,
    7.2324e-03, -4.2094e-02],
    ...,
    [-2.8452e-03, -2.6730e-03, 1.2535e+00, ..., -9.4729e-03,
    2.3407e-02, 3.2278e-02],
    [ 4.8053e-03, -3.7637e-02, -6.4973e-03, ..., 5.7681e-02,
    5.1689e-03, -7.2047e-04],
    [-4.1649e-03, -5.6664e-05, -1.5109e-03, ..., 5.5056e-03,
    1.3522e-02, -3.7987e-03]])
    However, the received data is:
    tensor([[-5.6711e-03, -5.5127e-04, 8.0800e-04, ..., 7.8006e-03,
    6.6992e-05, 9.6430e-03],
    [-1.6915e-03, 9.7358e-02, -6.7334e-03, ..., 2.1382e-02,
    -1.1618e-03, -1.0632e-02],
    [-3.0551e-03, -1.0185e-01, 6.1934e-02, ..., 1.1891e-02,
    7.2324e-03, -4.2094e-02],
    ...,
    [-1.4014e-02, 8.4079e-03, 1.0997e-03, ..., -1.3192e-02,
    8.4046e-04, 6.3276e-03],
    [ 1.0549e-02, -6.9067e-01, 5.1802e-03, ..., 1.5776e-01,
    -6.1614e-03, 1.7734e-03],
    [ 1.1559e+00, 7.0570e-01, -5.2229e-03, ..., 4.9758e-03,
    -2.0392e-03, 5.7419e-01]], device='cuda:0')
    The beginning part is identical, but there are differences at the end.
    I haven't found the reason, if I add a time.sleep(1) between shard sending and parity sending, this problem would be solved:
    for shard in shard_list:
        shard_send_requests.append(dist.isend(shard, src=...))
    time.sleep(1)
    for parity in parity_list:
        parity_send_requests.append(dist.isend(parity, src=...))
    So, I guess that maybe it's because the sent data influenced each other
    I can change this code to:
    for shard in shard_list:
        req = dist.isend(shard, src=...)
        req.wait()
    for parity in parity_list:
        req = dist.isend(parity, src=...)
        req.wait()
        
    Namely execute wait() after generating each sending request rather than collecting them and executing them together.

"""


"""
Learning notes:

1. For dist.all_gather(), the elements in tensor_list will be in the order of ranks.

2. Remove an element from a list
   1. list.remove(element_val) removes the first matching value from the list.
   2. del list[index] remove an element at a specific index or slice from a list.

3. Get a list like [0, 1, 2, 3....]
   You can't just use range(), range() returns a range object, it can only be used in iteration.
   To get a list, you need to use list(range(...))

4. One node broadcast a tensor to other nodes. No matter the sending node or receiving node, they should all
   use broadcast method

import torch
import torch.distributed as dist

# Initialize the distributed environment (assumes nccl backend)

dist.init_process_group('nccl')

rank = dist.get_rank()  # Get the rank of the current process
tensor = torch.zeros(10)  # Initialize a tensor on each node

if rank == 0:
    # Node 0 updates the tensor and broadcasts it
​    tensor += 1
​    dist.broadcast(tensor=tensor, src=0)
else:
    # Nodes 1, 2, and 3 receive the broadcasted tensor
​    dist.broadcast(tensor=tensor, src=0)

print(f"Node {rank} received tensor: {tensor}")
"""


"""
Thinking notes:

1. With dist, I can only send tensors directly, which means I can directly send the values of model state_dict(),
   but to send keys I need to convert to tensors, that's a annoying process. 
   So I prefer not to send the keys. The keys have been already recorded in the config json files.
   So for every healthy node, I simply need to send every value tensor in the ckpt["model"] by sequence. 
   And the failed node receives the tensors correspondingly. To restore the model dict, I just need to read the json
   file.

2. So for healthy nodes, the logic would be:
       - Send all shards with isend to the failed node
           - Send all parities with isend to the failed node
           - Use all gather to obtain shards from other healthy nodes
           - Wait until the tensors have been sent to the failed node
           - Use broadcast to get the restored shard from failed node
           - Combine all the shards to a complete model_state using the config json
           For the failed node, the logic would be:
           - Get a tensor list for both shards and parities initialized according to the config json files
           - Receive shards and parities from the healthy nodes using irecv
           - Recover its own shards 
           - *Calculate its own parity using the recovered shards (No need in this experiment)
           - Broadcast the shards to the healthy
           - Combine all the shards to a complete model_state using the config json

   return the recovered model_state_dict
"""
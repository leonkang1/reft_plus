import torch
import torch.optim as optim
import mmap
import numpy as np
import time
import os
def zero_copy_save(model, optimizer, model_mmap, opt_mmap):
    # 获取模型和优化器的权重存储对象
    model_storage = model.state_dict()['weight'].storage()
    opt_storage = optimizer.state_dict()['state'][list(optimizer.state_dict()['state'].keys())[0]]['momentum_buffer'].storage()

    # 内存映射文件用于存储模型权重
    model_mmapped = mmap.mmap(model_mmap.fileno(), model_storage.size() * 4)
    torch.FloatStorage.from_buffer(model_mmapped, byte_order='native').copy_(model_storage)

    # 内存映射文件用于存储优化器权重
    opt_mmapped = mmap.mmap(opt_mmap.fileno(), opt_storage.size() * 4)
    torch.FloatStorage.from_buffer(opt_mmapped, byte_order='native').copy_(opt_storage)

    model_mmapped.flush()
    opt_mmapped.flush()

def zero_copy_load(model, optimizer, model_mmap, opt_mmap):
    model_weight = torch.nn.Parameter(torch.Tensor(model.in_features, model.out_features))
    model_storage = model_weight.storage()

    model_mmapped = mmap.mmap(model_mmap.fileno(), model_storage.size() * 4)
    model_storage.copy_(torch.FloatStorage.from_buffer(model_mmapped))
    model.load_state_dict({'weight': model_weight})

    # Assume we know the shape of the optimizer momentum buffer
    opt_momentum = torch.Tensor(model.in_features, model.out_features)
    opt_storage = opt_momentum.storage()

    opt_mmapped = mmap.mmap(opt_mmap.fileno(), opt_storage.size() * 4)
    opt_storage.copy_(torch.FloatStorage.from_buffer(opt_mmapped))

    # Assume the optimizer state is empty and only contains momentum buffer for the weight
    optimizer.load_state_dict({
        'state': {0: {'momentum_buffer': opt_momentum}},
        'param_groups': optimizer.state_dict()['param_groups']
    })

# # 创建模型和优化器
# model = torch.nn.Linear(10, 10)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# # 零拷贝保存
# with open('model.mmap', 'wb+') as model_mmap, open('optimizer.mmap', 'wb+') as opt_mmap:
#     model_mmap.truncate(10 * 10 * 4)
#     opt_mmap.truncate(10 * 10 * 4)
#     zero_copy_save(model, optimizer, model_mmap, opt_mmap)

# # 零拷贝加载
# with open('model.mmap', 'rb') as model_mmap, open('optimizer.mmap', 'rb') as opt_mmap:
#     zero_copy_load(model, optimizer, model_mmap, opt_mmap)

# import time
# import os

# 创建模型和优化器
model = torch.nn.Linear(10, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 测量使用torch.save的序列化时间
start_time = time.time()
torch.save(model.state_dict(), "model.pth")
torch.save(optimizer.state_dict(), "optimizer.pth")
torch_save_time = time.time() - start_time

# 测量使用torch.save的反序列化时间
start_time = time.time()
model.load_state_dict(torch.load("model.pth"))
optimizer.load_state_dict(torch.load("optimizer.pth"))
torch_load_time = time.time() - start_time

inputs = torch.randn(32, 10)
outputs = model(inputs)
loss = outputs.mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()


# 零拷贝保存和加载
with open('model.mmap', 'wb+') as model_mmap, open('optimizer.mmap', 'wb+') as opt_mmap:
    model_mmap.truncate(10 * 10 * 4)
    opt_mmap.truncate(10 * 10 * 4)

    # 测量零拷贝的序列化时间
    start_time = time.time()
    zero_copy_save(model, optimizer, model_mmap, opt_mmap)
    zero_copy_save_time = time.time() - start_time

with open('model.mmap', 'rb') as model_mmap, open('optimizer.mmap', 'rb') as opt_mmap:
    # 测量零拷贝的反序列化时间
    start_time = time.time()
    zero_copy_load(model, optimizer, model_mmap, opt_mmap)
    zero_copy_load_time = time.time() - start_time

print(f"torch.save serialization time: {torch_save_time:.6f} seconds")
print(f"torch.load deserialization time: {torch_load_time:.6f} seconds")
print(f"Zero-copy serialization time: {zero_copy_save_time:.6f} seconds")
print(f"Zero-copy deserialization time: {zero_copy_load_time:.6f} seconds")

# Clean up
os.remove("model.pth")
os.remove("optimizer.pth")
os.remove("model.mmap")
os.remove("optimizer.mmap")

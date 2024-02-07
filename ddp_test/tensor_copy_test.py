import torch
from torch.profiler import profile, record_function, ProfilerActivity

def heavy_computation_on_gpu(tensor, repetitions=10):
    """在GPU上执行重复乘法计算来模拟重负载"""
    for _ in range(repetitions):
        tensor = tensor * tensor
    return tensor

def profile_copy_and_computation(non_blocking_copy, export_filename):
    # 确保PyTorch可以使用GPU
    assert torch.cuda.is_available()

    # 创建一个较大的张量
    large_tensor = torch.randn(10000, 10000, device='cuda')  # 调整大小以匹配内存容量
    large_tensor_cpu = torch.empty_like(large_tensor, device='cpu')
    result2 = heavy_computation_on_gpu(large_tensor)
    result3 = heavy_computation_on_gpu(large_tensor)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("copy_to_gpu"):
            # 将张量复制到GPU上，根据参数选择是否使用非阻塞复制
            large_tensor_cpu.copy_(large_tensor, non_blocking=non_blocking_copy)
        with record_function("heavy_computation_on_gpu"):
            # 在GPU上执行重负载计算
            result = heavy_computation_on_gpu(large_tensor)

    # 导出分析结果为Chrome Tracing兼容的JSON文件
    prof.export_chrome_trace(export_filename)


export_filename_blocking = "profiler_blocking.json"
print("\nProfiling with non_blocking=False")
profile_copy_and_computation(non_blocking_copy=False, export_filename=export_filename_blocking)

export_filename_non_blocking = "profiler_non_blocking.json"
print("Profiling with non_blocking=True")
profile_copy_and_computation(non_blocking_copy=True, export_filename=export_filename_non_blocking)

print(f"Profiler results exported to {export_filename_non_blocking} and {export_filename_blocking}.")
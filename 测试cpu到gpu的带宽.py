import torch
import time

# 创建一个大的张量
x = torch.randn(10000, 10000, 10)
print(f"{x.size()=}")

# 记录开始时间
start_time = time.time()

# 将数据从CPU复制到GPU
gpu = x.to('cuda')
del x

# 计算复制时间
cpu_to_gpu_time = time.time() - start_time

# 记录开始时间
start_time = time.time()

# 将数据从GPU复制回CPU
cpu = gpu.to('cpu')

# 计算复制时间
gpu_to_cpu_time = time.time() - start_time

print(f'CPU to GPU time: {cpu_to_gpu_time:.6f} seconds')
print(f'GPU to CPU time: {gpu_to_cpu_time:.6f} seconds')

input("回车继续")
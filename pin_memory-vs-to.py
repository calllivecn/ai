import torch
import time

# 从硬盘读取数据
#data = torch.randn(1000000, device='cpu')
data = torch.randn(3840*2160*3, device='cpu')

# 使用 tensor.pin_memory()
start_time = time.time()
data_pin = data.pin_memory()
data = data_pin.cuda()
# 在 GPU 上进行计算
data_pin *= 2
data_result = data_pin.cpu()
end_time = time.time()
print("tensor.pin_memory() time:", end_time - start_time, data_result.dtype, data_result.device)

# 直接使用 tensor.to("cuda")
start_time = time.time()
data = data.to('cuda')
print("tensor.to('cuda')", data.dtype, data.device)
# 在 GPU 上进行计算
data = data * 2
data = data.cpu()
end_time = time.time()
print("tensor.to('cuda') time:", end_time - start_time, data.dtype, data.device)


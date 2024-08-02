import time

import torch
from torch import profiler

def timeit(func):
    def wrap(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} 耗时：{t2-t1}")

        return result

    return wrap


profile = profiler.profile(
    schedule=profiler.schedule(wait=0, warmup=1, active=1, repeat=1),
    # on_trace_ready=profiler.tensorboard_trace_handler('/tmp/trace_prof'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    )


# 创建一个大的张量
x = torch.randn(10000, 1000, 10)
print(f"{x.size()=}")

gpu = None

# 将数据从CPU复制到GPU
@timeit
def cpu2gpu(count):
    for i in range(count):
        # x.pin_memory()
        gpu_x = x.to('cuda')
    
    global gpu
    gpu = gpu_x


# 将数据从GPU复制回CPU
@timeit
def gpu2cpu(count):
    for i in range(count):
        # cpu = gpu.to('cpu')
        cpu = gpu.cpu()


profile.start()
cpu2gpu(10)
gpu2cpu(10)

profile.stop()
profile.export_memory_timeline("/tmp/memory_timeline.json")
profile.export_chrome_trace("/tmp/chrome.json")



input("回车继续")

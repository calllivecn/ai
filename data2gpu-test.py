#!/usr/bin/env python3
# coding=utf-8
# date 2023-05-06 10:14:57
# author calllivecn <calllivecn@outlook.com>

# 测试 把数据移动GPU的性能消耗。

import sys

import torch


if not torch.cuda.is_available():
    print(f"当前不能使用cuda")
    sys.exit(0)

# torch.set_num_threads(4)

DEV = torch.device("cuda")

size2G = torch.randint(1, 100, size=(1024, 1024, 200), dtype=torch.float32, device=DEV)
y = torch.randint(1, 100, size=(200, 1024), dtype=torch.float32, device=DEV)

# cuda = size2G.to(DEV)
cuda = size2G
for i in range(1000):
    y.matmul(cuda)
    # del cuda
    # torch.cuda.empty_cache()


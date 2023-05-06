#!/usr/bin/env python3
# coding=utf-8
# date 2023-05-05 09:06:54
# author calllivecn <c-all@qq.com>

import torch

from torch import (
    nn,
    optim,
)

from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
    Sampler,
)


# import common
from common import (
    showmodel,
    showmodel_short,
)



# 需要拟合的函数
def func1(x):
    y = x * 2 + 3
    return y

def target_func(x, size):
    output = torch.randn(x, size, device=DEVICE)
    return output


class MyData(Dataset):

    def __init__(self, input_size, data_length) -> None:
        super().__init__()

        self.len = data_length

        self.batchsize = 1000

        # 这样就不能指定num_works != 0 了
        # self.data = torch.randn(length, size, device=DEVICE)
        self.data = torch.randn(self.batchsize, input_size)


    def __getitem__(self, index):
        c, p = divmod(index, self.batchsize)
        return self.data[p]
    
    def __len__(self):
        return self.len
    

    def parallel(self):
        pass


class MyDataIter(IterableDataset):

    def __init__(self, input_size, data_length) -> None:
        super().__init__()

        self.len = data_length

        self.batchsize = 150

        # 这样就不能指定num_works != 0 了
        # self.data = torch.randn(self.batchsize, input_size, device=DEVICE)

        # iter dataset 好像是一次返回
        self.data = torch.randn(input_size)


    def __iter__(self):
        for i in range(self.len):
            yield self.data



class Compute(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.Sequential1 = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Linear(1000, 2000),
            nn.Linear(2000, 4000),
            # nn.Conv1d(2, 2, kernel_size=3, padding=1),
            # nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.Linear(2000, 1000),
            nn.Linear(1000, output_dim),
        )

        """
        self.Sequential1 = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        """


    def forward(self, x):
        x = self.Sequential1(x)
        x = torch.relu(x)
        return x



input_size = 100
output_size = 10

batch_size = 200
data_size = 1000000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cuda"):
    torch.set_default_dtype(torch.float16)

elif DEVICE == torch.device("cpu"):
    print(f"设置 cpu 线程, pytorch 默认使用cpu * 1/2")
    # torch.set_num_threads(3)

print("走的:", DEVICE)

randn_loader = DataLoader(
    # dataset=MyData(input_size, data_size),
    dataset=MyDataIter(input_size, data_size),
    batch_size=batch_size,
    # num_workers=4,
    # pin_memory=True,
    # shuffle=True,
    )


model = Compute(input_size, output_size)
# model = torch.compile(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(DEVICE)
print(f"{model=}")

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion.to(DEVICE)

i = 0
for data in randn_loader:
    data = data.to(DEVICE)
    # target = func1(data)
    target = target_func(len(data), output_size)

    output = model(data)

    if i == 0:
        # print(f"{data=}")
        # print(f"{target=}")
        print(f"{len(data)=} {len(output)=}") # 输入的数据量和输出的是一样的
        print("Outside: input size", f"{data.size()=}", "output_size", f"{output.size()}", "y", f"{target.size()=}")
        showmodel_short(model)
        input("回车继续")

    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    i += 1
    if i % 100 == 0:
        print(f"epoch: {i*batch_size}")


showmodel(model)



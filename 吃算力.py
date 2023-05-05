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
    DataLoader,
)


# import common
from common import (
    showmodel,
    showmodel_short,
)


torch.set_default_dtype(torch.float16)


class MyData(Dataset):

    def __init__(self, size, length) -> None:
        super().__init__()

        self.len = length
        self.data = torch.randn(length, size, device=DEVICE)


    def __getitem__(self, index):
        return self.data[index]
    

    def __len__(self):
        return self.len
    

    def parallel(self):
        pass


class Compute(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.Sequential1 = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Linear(1000, 2000),
            nn.Linear(2000, 4000),
            # nn.Conv2d(2, 2, kernel_size=3,),
            # nn.ReLU(),
            nn.Linear(4000, 2000),
            nn.Linear(2000, output_dim),
        )


    def forward(self, x):
        x = self.Sequential1(x)
        x = torch.relu(x)
        return x



batch_size = 100

input_size = 500
output_size = 500

data_size = 1000000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

randn_loader = DataLoader(
    dataset=MyData(input_size, data_size),
    batch_size=batch_size,
    # num_workers=8, # 默认：0 当前cpu数
    # pin_memory=True,
    # pin_memory_device=DEVICE,
    # shuffle=True,
    )


model = Compute(input_size, output_size)
model = torch.compile(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion.to(DEVICE)

i = 0
for data in randn_loader:
    # data = data.to(DEVICE)
    output = model(data)
    optimizer.zero_grad()
    # print(f"{output.size()=}  {input.size()=}")
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()

    showmodel_short(model)
    i += 1
    print(f"epoch: {i}")




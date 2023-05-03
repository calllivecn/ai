#!/usr/bin/env py3
#coding=utf-8
# date 2023-05-02 17:51:55
# author calllivecn <c-all@qq.com>


import sys

import torch
from torch import (
    nn,
)
from torch.nn import (
    functional as F,
)

def f(x):
    return x * 2 + 3



class LinearReg(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
    

    def forward(self, x):
        out1 = self.linear1(x)
        return out1


def showmodel(model):
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")


model = LinearReg(1,1)

SEP="\n" + "="*40 + "\n"

def train():

    x_values = torch.randint(0, 10, size=(1000, 1), dtype=torch.float32)
    y_values = f(x_values)


    learing_rate = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate) #, weight_decay=0.005)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    # def criterion(values, labels):
        # return values - labels

    # 训练
    epoch = 0 
    epoch_count = 100
    for epoch in range(1000):
    # for i, x in enumerate(x_values):
    # while True:

        # 梯度要清零
        optimizer.zero_grad()

        # 向前传播
        outputs = model(x_values)
        # outputs = model(x)

        # 计算损失
        loss = criterion(outputs, y_values)
        # loss = F.mse_loss(outputs, y_values)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        if epoch % epoch_count == 0:
            print("+"*20, f"epoch: {epoch}", "+"*20)
            print(f"{loss=}", f"{optimizer=}", sep=SEP, end=SEP)
            showmodel(model)

        epoch += 1
    
    print(model)
    showmodel(model)

    # 保存模型
    torch.save(model.state_dict(), "model.pkl")


# 测试, 测试平均误差
def valid():

    # x_valid = torch.Tensor(np.random.randint(0, 100, size=(100,1)))
    x_valid = torch.randint(0, 100, size=(100,1), dtype=torch.float32)

    y_valid = f(x_valid)

    # 加载模型
    model.load_state_dict(torch.load("model.pkl"))

    with torch.no_grad():
        y_ = model(x_valid)


    # 求平均误差
    res = torch.abs(y_ - y_valid)
    print(x_valid[:10], y_[:10], res[:10], sep="\n")
    mean = torch.mean(res)

    mseloss = nn.MSELoss()
    mean_mse = mseloss(y_, y_valid)

    print(f"平均误差：{mean=} {mean_mse=}",  torch.mean(y_valid), torch.mean(y_))
    showmodel(model)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        valid()
    elif sys.argv[1] == "train":
        train()
    else:
        valid()
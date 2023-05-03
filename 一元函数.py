#!/usr/bin/env py3
#coding=utf-8
# date 2023-05-02 17:51:55
# author calllivecn <c-all@qq.com>


import sys

import numpy as np

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
        # self.linear2 = nn.Linear(2, output_dim)
    

    def forward(self, x):
        out1 = self.linear1(x)
        # out2 = F.relu(self.linear2(out1))
        return out1


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


def showmodel(model):
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")


model = LinearReg(1,1)

"""
if device == torch.device("cuda"):
    print("走了CUDA")
    model.to(device)
else:
    print("走了CPU")

"""
SEP="\n" + "="*40 + "\n"

def train():

    x_values = torch.randint(0, 1, size=(1000, 1), dtype=torch.float32)
    # print(x_values[:10]);exit(0)

    # y = x * 2 + 3
    y_values = f(x_values)


    learing_rate = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate, weight_decay=0.005)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()

    # def criterion(values, labels):
        # return torch.Tensor(values - labels)

    # 训练
    epoch = 0 
    # for epoch in range(10000):
    # for i, x in enumerate(x_values):
    epoch_print = 1
    while True:

        # 梯度要清零
        optimizer.zero_grad()

        # 向前传播
        outputs = model(x_values)
        # outputs = model(x)

        # 计算损失
        loss = criterion(outputs, y_values)
        # if loss > 10:
            # loss = torch.Tensor(10, grad_fn=loss.grad_fn, requires_grad=True, dtype=torch.float32)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        if epoch % epoch_print == 0:
            print(f"{type(loss)=}", f"{loss=}", f"{optimizer=}", f"epoch: {epoch}", sep=SEP, end=SEP)
            showmodel(model)
            n  = input(f"可以输入数字为每epoch轮输出一次：")
            if n == "/quit":
                sys.exit(0)
            
            elif n == "/done":
                break

            elif isinstance(n, str):
                try:
                    n = int(n)
                    epoch_print = n
                except Exception:
                    pass

        
        # 当训练集准确度小于0.0001时结束训练
        # if loss.item() <= 0.0001:
            # print(f"epoch: {epoch}, loss: {loss}")
            # break

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
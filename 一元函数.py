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

SEP="\n" + "="*40 + "\n"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DTYPE=torch.float32


def f(x):
    return x * 2 + 3



class LinearReg(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, dtype=DTYPE)
        # self.linear2 = nn.Linear(2, output_dim)
    

    def forward(self, x):
        out1 = self.linear1(x)
        # out2 = F.relu(self.linear2(out1))
        return out1



def showmodel(model):
    print("="*20, "查看模型", "="*20)
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")




def train():

    model = LinearReg(1,1)

    if device == torch.device("cuda"):
        print("走了CUDA")
        model.to(device)
    else:
        print("走了CPU")

    # input("回车继续")

    learing_rate = 0.01

    optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate, momentum=0.9) #, weight_decay=0.005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    criterion.to(device)

    # def criterion(values, labels):
        # return torch.Tensor(values - labels)

    # 训练
    epoch = 0 
    epoch_count = 10
    # for epoch in range(100):
    # for i, x in enumerate(x_values):
    while True:

        # 每次batch_size小点，效果不错。
        x_values = torch.randint(0, 10, size=(100, 1))
        x_values = x_values.to(device, dtype=DTYPE)
        # print(x_values[:10]);exit(0)

        # y = x * 2 + 3
        y_values = f(x_values)


        # 梯度要清零
        optimizer.zero_grad()

        # 向前传播
        outputs = model(x_values)
        # outputs = model(x)

        # 计算损失
        loss = criterion(outputs, y_values)

        # 反向传播
        loss.backward()

        # 更新权重
        optimizer.step()

        if epoch % epoch_count == 0:
            print(f"{loss=} {optimizer=} {epoch=}")
            showmodel(model)
            """
            n  = input(f"可以输入数字为每epoch轮输出一次：")
            if n == "/quit":
                sys.exit(0)
            
            elif n == "/done":
                break

            elif isinstance(n, str):
                try:
                    n = int(n)
                    epoch_count = n
                except Exception:
                    pass
            """
        
        # 当训练集准确度小于0.0001时结束训练
        if loss.item() <= 1e-6:
            print(f"epoch: {epoch}, loss: {loss}")
            break

        epoch += 1
    
    print(model)
    showmodel(model)

    # 保存模型
    torch.save(model.state_dict(), "model.pkl")



# 测试, 测试平均误差
def valid():


    model = LinearReg(1,1)

    if device == torch.device("cuda"):
        print("走了CUDA")
        model.to(device)
    else:
        print("走了CPU")

    # x_valid = torch.Tensor(np.random.randint(0, 100, size=(100,1)))
    x_valid = torch.randint(0, 100, size=(100,1))
    x_valid = x_valid.to(device, dtype=DTYPE)

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


def main():
    import argparse

    parse = argparse.ArgumentParser()
    groups = parse.add_mutually_exclusive_group()
    groups.add_argument("--train", action="store_true", help="训练")
    groups.add_argument("--valid", action="store_true", help="训练")

    parse.add_argument("--parse", action="store_true", help=argparse.SUPPRESS)
    args = parse.parse_args()

    if args.parse:
        print(args)
        sys.exit(0)

    if args.train:
        train()
    else:
        valid()


if __name__ == "__main__":
    main()
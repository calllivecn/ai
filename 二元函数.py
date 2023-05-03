
import torch
from torch import (
    nn,
    optim,
)

def showmodel(model):
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")


# 定义函数
def f(x1, x2):
    return x1 * 2 * x2 + 3

# 定义数据集
x = torch.randint(0, 1, size=(1000, 2), dtype=torch.float32)
y = f(x[:, 0], x[:, 1])

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

net = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    optimizer.zero_grad()
    output = net(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("="*40)
        showmodel(net)



print("预测结果：", net(torch.tensor([10.0, 20.0])), "实际结果：", f(10.0, 20.0))

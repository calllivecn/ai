
import torch
from torch import (
    nn,
    optim,
)

def showmodel(model):
    print("="*20, "查看模型", "="*20)
    for name, param in model.named_parameters():
        print(f"{name} | {param.size()} | {param[:2]}")


# 定义函数
def f(x1, x2):
    return x1 * 2 + x2 * 5 + 3


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("走的：", DEV)

net = Net()
net.to(DEV)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

criterion.to(DEV)

# 训练模型
for epoch in range(1, 1000 + 1):

    # 定义训练数据集
    x = torch.randint(0, 10, size=(1000, 2), dtype=torch.float32)
    x = x.to(DEV)
    y = f(x[:, 0], x[:, 1])

    optimizer.zero_grad()
    output = net(x)
    # loss = criterion(output, y)
    loss = criterion(output.squeeze(), y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        showmodel(net)
        print("+"*20, "查看训练过程", "+"*20)
        print(f"{epoch=} {loss=} {optimizer=} ")
    
    # if loss <= 1e-6:
        # break

print(f"训练完成")

showmodel(net)


x_test = torch.tensor([10.0, 20.0], dtype=torch.float32)
x_test = x_test.to(DEV)
print("预测结果：", net(x_test), "实际结果：", f(10.0, 20.0))

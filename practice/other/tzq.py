import torch
from torch import nn
from numpy import array


class lstm(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 4x1x3
        x, _ = self.layer1(x)   # 4x1x4
        s, b, h = x.size()
        x = x.view(s * b, h)    # 4x4
        x = self.layer2(x)      # 4x1
        x = x.view(s, b, -1)    # 4x1x1
        return x


# 定义训练样本
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# 默认格式(序列长度, batchsize, 输入特征数) ==> (4, 1, 3)
X = X.reshape(-1, 1, 3)
X = torch.from_numpy(X).float()
y = y.reshape(-1, 1, 1)     # 4x1x1
y = torch.from_numpy(y).float()
model = lstm(3, 4, 1, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
criterion = nn.MSELoss()
# criterion = nn.L1Loss(reduction='sum')
# criterion = nn.CrossEntropyLoss()
model = model.train()
for e in range(1000):
    var_x = X
    # print(var_x)
    var_y = y
    # 前向传播  4x1x3
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}, Result_1:{}'
              .format(e + 1, loss.item(), out))


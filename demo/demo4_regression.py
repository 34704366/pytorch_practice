#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorch_practice 
@File    ：demo4_regression.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/3/5 21:24 
'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt



class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super().__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output) # output layer
    def forward(self, x):
        # 将x放入隐藏层hidden()中，然后经过relu函数，再经过predict层
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    # unsqueeze是将一维数据转换为二维数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.__pow__(2) + 0.2 * torch.rand(x.size())

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    net = Net(n_feature=1, n_hidden=10, n_output=1)
    print(net)

    plt.ion()  # something about plotting

    # lr是学习效率
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    # 损失函数指定为均方误差
    loss_func = torch.nn.MSELoss()

    for t in range(200):
        # 计算预测值
        prediction = net(x)
        # 计算损失值
        loss = loss_func(prediction, y)

        # 先将所有参数的梯度全部降为0
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        optimizer.step()

        if t % 5 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()
















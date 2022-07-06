#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorch_practice 
@File    ：demo5_classification.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/3/5 21:59 
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
    n_data = torch.ones(100, 2)
    x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
    y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
    x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
    y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
    x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
    y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap="RdYlGn")
    # plt.show()
    #
    # print(y0)
    # print(y1)
    # print(y)

    net = Net(n_feature=2, n_hidden=10, n_output=2)
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()   # CrossEntropyLoss是交叉熵损失函数，数值越小说明和期望结果越接近
    # loss_func = F.cross_entropy()


    plt.ion()   # something about plotting


    for t in range(100):
        #计算预测值
        out = net(x)

        # 计算误差
        loss = loss_func(out, y)

        # 清除上一批的梯度
        optimizer.zero_grad()

        # 误差反向传递
        loss.backward()

        # 把梯度施加到神经网络中（进行一次优化）
        optimizer.step()

        # 可视化动画
        if t % 2 == 0:
            # plot and show learning process
            plt.cla()
            out = torch.max(out, 1)[1]
            pred_y = out.data.numpy()
            target_y = y.data.numpy()
            plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
            accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
            plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

    plt.ioff()
    plt.show()











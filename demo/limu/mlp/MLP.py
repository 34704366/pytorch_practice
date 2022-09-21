#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：machine_learning 
@File    ：MLP.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/9/19 21:56 
'''
import torch
import torch.nn as nn

x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
print(x, x.grad)
y = x * 2
while y.norm() < 1000:
    y = y * 2
    print(y, y.norm())

print('乘法之后',y)

y.backward(torch.ones_like(y))
print(x, x.grad)

# 将x的grad清除
x.grad.data.zero_()

print(x, x.grad)


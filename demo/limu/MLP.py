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
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
image = d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pytorch_practice 
@File    ：demo2.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/3/5 17:19 
'''

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)
print(variable)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out, v_out)
v_out.backward()
print(variable.grad)


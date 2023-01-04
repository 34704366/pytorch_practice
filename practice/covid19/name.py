#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：remote_practice 
@File    ：name.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/10/31 17:20 
'''

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
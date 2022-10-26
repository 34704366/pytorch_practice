#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：remote_practice 
@File    ：name.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/10/19 17:23 
'''
import math

# import numpy as np
#
# data = np.load('../dataset/mat.npy')
#
# print(data[1:10, 1:10])


# 十进制数转换成任意进制数
# 输入十进制数并分离整数小数
print("请输入待转换的十进制数:")
a = float(input())
print(a)
int_part_a = int(a)
float_part_a = a - int_part_a
print(f"待转换的十进制数的整数部分是:{int_part_a}")
print(f"待转换的十进制数的小数部分是:{float_part_a}")

# 输入目标进制
print("请输入要转换成的进制:(2<=目标进制<10)")
base = eval(input())

# 整数部分转换进制
z = int_part_a
m = ""
while z > 0:
    m = m + str(int(z % base))
    z = z // base

# 小数部分转换进制
x = float_part_a
n = ""
while x > 0:
    n = n + str(int(x * base))
    x = x * base - int(x * base)
    if len(n) >= 8:
        n = n[0:8]
        break
# 输出结果s
print(f"({m}.{n}){base}")

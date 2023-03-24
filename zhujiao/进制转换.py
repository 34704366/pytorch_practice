#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：remote_practice 
@File    ：进制转换.py
@IDE     ：PyCharm 
@Author  ：王晨昊
@Date    ：2022/10/30 15:54 
'''
# 输入
number = float(input("输入待转换的十进制数:"))
int_part = int(number)
float_part = number - int_part
print(f"整数部分是:{int_part}")
print(f"小数部分是:{float_part}")

# 输入目标进制
base = int(float(input("输入要转换成的进制:")))

# 整数部分转换进制
remainder = int_part
int_result = ""
while remainder > 0:
    int_result = str(int(remainder % base)) + int_result
    remainder = remainder // base

# 小数部分转换进制
remain_number = 8    # 保留几位小数
remainder = float_part
float_result = ""
while remainder > 0:
    float_result = float_result + str(int(remainder * base))
    remainder = remainder * base - int(remainder * base)
    if len(float_result) >= remain_number:
        break
# 输出结果
print(f"({int_result}.{float_result}){base}")

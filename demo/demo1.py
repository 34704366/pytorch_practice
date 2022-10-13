import torch
import numpy as np


np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
# print(np_data)
# print(torch_data)
# print(tensor2array)

data = [-1, -2 ,1 ,2]
tensor = torch.FloatTensor(data)    # 32bit
# print(tensor)
# print(torch.abs(tensor))
# print(torch.sin(tensor))
# print(torch.mean(tensor))

data = [[1, 2], [5, 6]]
tensor = torch.FloatTensor(data)
print(np.matmul(data, data))    # 和data.dot(data)同效果
print(torch.mm(tensor, tensor))    # 和tensor.dot(tensor)同效果









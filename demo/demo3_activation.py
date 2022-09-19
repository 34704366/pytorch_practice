import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.linspace(-5, 5, 200)
# 使用mlt画图的时候，torch数据是不能直接被识别的，还是需要numpy格式的数据
x_np = x.data.numpy()

y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
# y_softmax = torch.softmax(x).data.numpy()

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label="relu")
plt.plot(x_np, y_sigmoid, c='blue', label="relu")
plt.plot(x_np, y_tanh, c='yellow', label="relu")
# plt.plot(x_np, y_softmax, c='green', label="relu")

plt.ylim((-1,5))
plt.legend(loc="best")


plt.show()
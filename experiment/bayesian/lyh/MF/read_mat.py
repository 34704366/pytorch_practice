import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = np.load('./mat.npy', allow_pickle = True)
print(file.shape)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("./dataset/Delaware.csv")
# df = df[df.REGION == "Delaware"]

df_use = df[["REGION", "YEAR", "WEEK", "ili_ratio"]]
df_use.index = range(len(df_use))

df_use = (df_use.loc[:, ["ili_ratio"]])  # 只用ili_ratio
print(df_use)

y = np.array(df_use['ili_ratio'])
y = y[:]

plt.plot(range(y.__len__()), y, c='blue', marker='*', ms=1, alpha=0.75, label='true')
plt.show()
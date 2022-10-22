import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

data = pd.read_csv("../泰迪杯预测2-min.csv")
data["Month"] = pd.to_datetime(data['Month'])

print(data)

ROWS = 16
COLS = 93


def getTime(timestr = None):
    year = int(timestr[0:4])
    month = int(timestr[5:7])
    days = int(timestr[8:10])
    hours = int(timestr[11:13])
    minutes = int(timestr[14:16])
    return year, month, days, hours, minutes


mat = [[0]*COLS for _ in range(ROWS)]

for i,rows in data.iterrows():
    year, month, day, hours, mins = getTime(str(rows["Month"]))
    r = (year - 2018) * 4 + (month - 1) // 3
    c = ((month % 3 - 1) % 3) * 31 + day - 1
    mat[r][c] = rows["value"]

np.save('./ans-min.npy', mat)
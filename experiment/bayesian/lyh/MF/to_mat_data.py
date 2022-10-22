import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

data = pd.read_csv("../泰迪杯数据.csv")
data["Month"] = pd.to_datetime(data['Month'])


ROWS = 16
COLS = 8928


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
    c = ((month % 3 - 1) % 3) * 31 * 96 + (day - 1) * 96 + (hours * 60 + mins) // 15
    mat[r][c] = rows["value"]

np.save('./mat.npy', mat)
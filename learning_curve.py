import matplotlib.pyplot as plt
import numpy as np

lines = []
with open('result.txt', 'r') as history:
    lines = history.readlines()

mins = []
maxs = []
avgs = []
for line in lines:
    line_sep = line.split(" ")
    mins.append(float(line_sep[0]))
    avgs.append(float(line_sep[1]))
    maxs.append(float(line_sep[2]))
x = np.arange(1, len(lines) + 1, 1)
plt.plot(x, mins, label='min')
plt.plot(x, avgs, label='average')
plt.plot(x, maxs, label='max')
plt.legend()
plt.show()

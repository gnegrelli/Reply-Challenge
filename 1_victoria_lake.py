

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

pullData = open("1_victoria_lake.txt", "r").read()
dataList = pullData.split("\n")

first_row = dataList[0].split(" ")

dic = {}

dic['map_size'] = (float(first_row[0]), float(first_row[1]))
dic['Customer_HQ'] = float(first_row[2])
dic['Reply_Offices'] = float(first_row[3])

HQ = []

for i in range(int(dic['Customer_HQ'])):
    row_info = dataList[i+1].split(" ")
    HQ.append([(float(row_info[0]), float(row_info[1])), float(row_info[2])])

# print HQ[0][0]

Map = np.array([])

for eachLine in dataList[int(dic['Customer_HQ'])+1:]:
    line = np.array([])
    for eachChar in eachLine:
        if line.size < 50:
            if eachChar == '~':
                line = np.hstack((line, np.array(800)))
            elif eachChar == '*':
                line = np.hstack((line, np.array(200)))
            elif eachChar == '+':
                line = np.hstack((line, np.array(150)))
            elif eachChar == 'X':
                line = np.hstack((line, np.array(120)))
            elif eachChar == '_':
                line = np.hstack((line, np.array(100)))
            elif eachChar == 'H':
                line = np.hstack((line, np.array(70)))
            elif eachChar == 'T':
                line = np.hstack((line, np.array(50)))
            else:
                line = np.hstack((line, np.array(1000)))
    try:
        Map = np.vstack((Map, line))
    except:
        if line.size > 0:
            Map = line

heatmap = 1 - Map/800

plt.imshow(heatmap, cmap='hot', interpolation='nearest')

fig = plt.figure()
ax = fig.gca(projection='3d')

A, B = np.meshgrid(range(50), range(50))
surf = ax.plot_surface(A, B, Map, linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

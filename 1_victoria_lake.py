

import numpy as np
import matplotlib.pyplot as plt

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

# print np.hstack((Map, 1.))

for eachLine in dataList[int(dic['Customer_HQ'])+1:]:
    line = np.array([])
    for eachChar in eachLine:
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
            line = np.hstack((line, np.array(2000)))

    # print line
    try:
        Map = np.vstack((Map, line))
    except:
        if line.size > 0:
            Map = line

Map = 1 - Map/2000

plt.imshow(Map, cmap='hot', interpolation='nearest')
plt.show()

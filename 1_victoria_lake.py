

import numpy as np

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

print HQ[0]


# for eachLine in dataList:
    # print eachLine

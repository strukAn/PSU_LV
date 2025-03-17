from msilib import datasizemask
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols = (1, 2, 3, 4, 5, 6), delimiter = ",", skiprows = 1)

print(f"ALL:\n\tMAX mpg: {data.max(axis = 0)[0]}\n\tMIN mpg: {data.min(axis = 0)[0]}\n\tAvg mpg: %.1f" % round(data.mean(axis = 0)[0], 1))

indices = []

index = 0
for row in data:
    if row[1] == 6.0:
        indices.append(index)
        
    index += 1
    
subData = data.take(indices, axis = 0)

print(f"6 CYL:\n\tMAX mpg: {subData.max(axis = 0)[0]}\n\tMIN mpg: {subData.min(axis = 0)[0]}\n\tAvg mpg: %.1f" % round(subData.mean(axis = 0)[0], 1))
    
for i in range(len(data)):
    plt.scatter(data[i][0], data[i][3], marker = "o", s = data[i][5] * 20)
    
plt.xlabel("mpg")
plt.ylabel("hp")
plt.show()
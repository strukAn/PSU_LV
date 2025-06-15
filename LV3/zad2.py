import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#1

data = pd.read_csv("mtcars.csv")

subData = data[data['cyl'].isin([4, 6, 8])]

plt.figure(figsize=(10, 5))
bars = plt.bar(subData['car'], subData['mpg'], color='skyblue')

plt.xticks(rotation=90, ha='right')

plt.xlabel("car")
plt.ylabel("mpg")

plt.show()

#2

subData = data[data['cyl'].isin([4, 6, 8])]

subData = [subData[subData['cyl'] == cyl]['wt'] for cyl in [4, 6, 8]]

plt.figure(figsize=(10, 5))
plt.boxplot(subData, labels=['4', '6', '8'])

plt.xlabel("cyl")
plt.ylabel("wt")

plt.show()

#3

subData = data.groupby('am')['mpg'].mean()

labels = ['Manual', 'Automatic']
values = [subData[0], subData[1]]

plt.bar(labels, values)

plt.xlabel("transmission")
plt.ylabel("mpg")

plt.show()

#4

manual = data[data['am'] == 0]
automatic = data[data['am'] == 1]

plt.figure(figsize=(10, 5))

plt.scatter(manual['hp'], manual['qsec'], color='blue', label='Manual')

plt.scatter(automatic['hp'], automatic['qsec'], color='red', label='Automatic')

plt.xlabel('hp')
plt.ylabel('qsec')

plt.legend()

plt.show()
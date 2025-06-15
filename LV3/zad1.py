import pandas as pd
import numpy as np

data = pd.read_csv("mtcars.csv", delimiter = ",")

data = data.sort_values(by = "mpg", axis = 0, ascending = True)

print(data)

subData = data.nsmallest(5, "mpg")
print(f"\nNajveci potrosaci:")
print(subData)

subData = data[data["cyl"] == 8]
subData = subData.nlargest(3, "mpg")
print(f"\nNajmanji potrosaci s 8 cilindara:")
print(subData)

subData = data[data["cyl"] == 6].filter(like = "mpg")
print(f"\nSrednja potrosnja automobila s 6 cilindara: {subData.mean().iloc[0]} mpg")

subData = data[data["cyl"] == 4]
subData = subData[subData["wt"] >= 2.000]
subData = subData[subData["wt"] <= 2.200].filter(like = "mpg")

print(f"\nSrednja potrosnja automobila s 4 cilindra mase izmedu 2000 i 2200 lbs: {subData.mean().iloc[0]}")

subData = data[data["am"] == 1]
print(f"\nAutomobila s rucnim mjenjacem: {subData.count().iloc[0]}")
subData = data[data["am"] == 0]
print(f"Automobila s automatskim mjenjacem: {subData.count().iloc[0]}")

subData = subData[subData["hp"] >= 100.0]
print(f"Automobila s automatskim mjenjacem i snagom preko 100hp: {subData.count().iloc[0]}\n")

for i, row in data.iterrows():
    name = row["car"]
    x = int(len(name) / 16)
    if(len(name) < 8):
        x -= 1
    
    for j in range(2 - x):
        name += "\t"
        
    print(name + str(round(row["wt"] * 1000 / 2.205, 2)) + "\tkg")
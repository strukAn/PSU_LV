dict1 = {}

with open("song.txt") as file:
    for line in file:
        words = line.strip().split()
        for word in words:
            word = word.strip()
            if word not in dict1:
                dict1[word] = 1
            else:
                dict1[word] += 1
    file.close()

for value in dict1:
    if(dict1[value] == 1):
        print(f"{value} : {dict1[value]}")
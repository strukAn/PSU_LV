print("Unesite ocijenu [0,1]: ")
while(True):
    try:
        x = float(input())
    except:
        print("Upis mora biti broj")
        continue
    if(x >= 0 and x <= 1):
        break
    else:
        print("Broj mora biti u intervalu [0,1]")
        print("Ponovno unesite ocijenu [0,1]: ")

if(x >= 0.9):
    print('A')
elif(x >= 0.8):
    print('B')
elif(x >= 0.7):
    print('C')
elif(x >= 0.6):
    print('D')
else:
    print('F')
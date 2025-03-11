nums = []
num = 0

while(True):
    text = input()
    text = text.strip()
    if(text == "DONE"):
        break

    try:
        num = int(text.strip())
    except:
        print("Unos mora biti broj")
        continue
    nums.append(num)

print(f"MAX: {max(nums)}\nMIN: {min(nums)}\nSrednja vrijednost: {sum(nums) / len(nums)}")
nums.sort()
print(nums)
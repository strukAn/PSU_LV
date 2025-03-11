def total_euro(hours , per_hour):
    print(f"Ukupno: {hours * per_hour} eura")

print("Radni sat: ")
hours = float(input())
print("eura/h: ")
wage = float(input())

total_euro(hours, wage)
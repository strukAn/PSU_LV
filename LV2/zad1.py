import numpy as np
import matplotlib.pyplot as plt

arr = np.array([[1, 2, 3, 3, 1], [1, 2, 2, 1, 1]])

plt.plot(arr[0], arr[1], "go--", linewidth = 3, markersize = 10)

plt.axis([0, 5, 0, 5])
plt.xlabel("x")
plt.ylabel("y")
plt.title("zad1")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def board(size: int, x_squares: int, y_squares: int) -> np.matrix:
    row = np.zeros( (size, size) )
    square = np.zeros( (size, size) )
    _square = np.ones( (size, size) ) * 255
    
    white = True
    for i in range(x_squares - 1):
        if white:
            row = np.hstack( (row, _square) )
        else:
            row = np.hstack( (row, square) )

        white = not white            

    _row = 255 - row
    img = row
    
    white = True
    for i in range(y_squares - 1):
        if white:
            img = np.vstack( (img, _row) )
        else:
            img = np.vstack( (img, row) )
            
        white = not white
    
    return img
    
img = board(250, 5, 5)

plt.figure()
plt.imshow(img, cmap = "gray", vmax = 255, vmin = 0)
plt.show()
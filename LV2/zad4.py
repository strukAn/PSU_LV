import numpy as np
import matplotlib.pyplot as plt

def board(size: int, x_squares: int, y_squares: int) -> np.matrix:
    row = np.zeros( (size, size) )
    
    white = True
    for i in range(x_squares - 1):
        if white:
            new = np.ones( (size, size) )
        else:
            new = np.zeros( (size, size) )

        white = not white            

        row = np.hstack( (row, new) )
    
    _row = 1 - row
    img = row
    
    white = True
    for i in range(y_squares - 1):
        if white:
            img = np.vstack( (img, _row) )
        else:
            img = np.vstack( (img, row) )
            
        white = not white

    
    plt.figure()

    plt.imshow(img, cmap = "gray", vmax = 1, vmin = 0)
    plt.show()
    

board(250, 10, 10)
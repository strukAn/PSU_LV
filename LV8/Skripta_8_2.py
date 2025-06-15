import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage import color
from tensorflow.keras import models, saving
import numpy as np

filename = 'test4.png'

# Ucitaj sliku
img_original = mpimg.imread(filename)  # Zamijeni 'test.png' s putanjom do svoje slike
img = color.rgb2gray(img_original)
img = resize(img, (28, 28))

# Prikazi sliku
plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.axis('off')  
# plt.show()

# Pripremi sliku - ulaz u mrezu
img = img.reshape(1, 28, 28, 1)
img = img.astype('float32')

# TODO: ucitaj izgradenu mrezu

model = saving.load_model('model.keras')

# TODO: napravi predikciju za ucitanu sliku pomocu mreze

pred = model.predict(img)

# TODO: ispis rezultat u terminal

pred_class = np.argmax(pred, axis=1)
print(f'Predicted class: {pred_class[0]}')

from tensorflow.keras import models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np

img = image_dataset_from_directory(
 directory='test',
 labels='inferred',
 label_mode='categorical',
 batch_size=1,
 image_size=(48, 48))

best_model = models.load_model('best_model.h5')

pred = best_model.predict(img)

pred_class = np.argmax(pred, axis=1)
print(f'Predicted class: {pred_class}')
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_ds = image_dataset_from_directory(
 directory='dataset/Train',
 labels='inferred',
 label_mode='categorical',
 batch_size=32,
 subset="training",
 seed=123,
 validation_split=0.2,
 image_size=(48, 48))

validation_ds = image_dataset_from_directory(
 directory='dataset/Train',
 labels='inferred',
 label_mode='categorical',
 batch_size=32,
 subset="validation",
 seed=123,
 validation_split=0.2,
 image_size=(48, 48))

test_ds = image_dataset_from_directory(
 directory='dataset/Test',
 labels='inferred',
 label_mode='categorical',
 batch_size=32,
 image_size=(48, 48))

# x_test = tf.concat([x for (x, y) in test_ds], axis = 0)
# x_train = tf.concat([x for (x, y) in train_ds], axis = 0)
# x_val = tf.concat([x for (x, y) in validation_ds], axis = 0)

y_test = tf.concat([y for (x, y) in test_ds], axis = 0)
# y_train = tf.concat([y for (x, y) in train_ds], axis = 0)
# y_val = tf.concat([y for (x, y) in validation_ds], axis = 0)

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 3), padding='same'),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.2),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.2),    

    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = callbacks.TensorBoard(log_dir='logs')
checkpoint_callback = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')

history = model.fit(train_ds, validation_data=validation_ds, epochs=3, batch_size=32,
                    callbacks=[tensorboard_callback, checkpoint_callback])

best_model = models.load_model('best_model.h5')
test_loss, test_acc = best_model.evaluate(test_ds)
print(f'Test accuracy: {test_acc}')

y_test_pred = best_model.predict(test_ds)
y_test_pred_classes = tf.argmax(y_test_pred, axis=1)

y_test = tf.argmax(y_test, axis=1)

cm_test = confusion_matrix(y_test, y_test_pred_classes)
plt.figure(figsize=(15, 10))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(43), yticklabels=np.arange(43))
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
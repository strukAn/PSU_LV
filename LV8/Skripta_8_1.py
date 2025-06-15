from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)

# TODO: strukturiraj konvolucijsku neuronsku mrezu

model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# TODO: definiraj callbacks

tensorboard_callback = callbacks.TensorBoard(log_dir='logs')
checkpoint_callback = callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')


# TODO: provedi treniranje mreze pomocu .fit()

history = model.fit(x_train_s, y_train_s, epochs=10, batch_size=32,
                    validation_split=0.1,
                    callbacks=[tensorboard_callback, checkpoint_callback])

model.save('model.keras')

#TODO: Ucitaj najbolji model

best_model = keras.models.load_model('best_model.h5')

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

train_loss, train_acc = best_model.evaluate(x_train_s, y_train_s)
print(f'Train accuracy: {train_acc}')

test_loss, test_acc = best_model.evaluate(x_test_s, y_test_s)
print(f'Test accuracy: {test_acc}')


# TODO: Prikazite matricu zabune na skupu podataka za testiranje

y_train_pred = best_model.predict(x_train_s)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

y_test_pred = best_model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

cm_train = confusion_matrix(y_train, y_train_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix for Training Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

cm_test = confusion_matrix(y_test, y_test_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix for Test Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
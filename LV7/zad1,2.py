import numpy as np
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, classification_report


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa

# for i in range(0, 5):
#     plt.imshow(x_train[int(int(x_train.size / x_train[0].size) / 2) + i], cmap='gray')
#     plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()

model = Sequential()
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()

model.compile(loss='categorical_crossentropy',
 optimizer='sgd',
 metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit()

model.fit(x_train_s, y_train_s, epochs=25, batch_size=32)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje

y_train_pred = model.predict(x_train_s)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

y_test_pred = model.predict(x_test_s)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

train_loss, train_acc = model.evaluate(x_train_s, y_train_s)
print(f'Train accuracy: {train_acc}')

test_loss, test_acc = model.evaluate(x_test_s, y_test_s)
print(f'Test accuracy: {test_acc}')

# TODO: Prikazite matricu zabune na skupu podataka za testiranje

cm_test = confusion_matrix(y_test, y_test_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix for Testing Set')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgra?ena mreza pogresno klasificirala

y_test_pred = keras.utils.to_categorical(y_test_pred, 10)

wrong_indices = np.where(y_test_pred_classes != y_test)[0]

for i in range(5):
    idx = wrong_indices[i]
    image = x_test[idx]
    true_label = y_test_s[idx]
    predicted_label = y_test_pred[idx]

    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.show()
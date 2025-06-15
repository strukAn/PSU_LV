import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, accuracy_score, classification_report

df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size = 0.8,train_size = 0.2, stratify=y)

scaler = skp.StandardScaler()
scaler.fit(X_train, y_train)
scaler.fit(X_test, y_test)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0',
'Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
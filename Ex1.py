import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loan_dataset = pd.read_csv(r"D:\FML_AI\CTAI-ML\ML-exercises\Ex1\Loan Modelling Thera Bank.csv")


X = loan_dataset.drop(['Personal Loan', 'ID'], axis=1)
X = (X - X.min()) / (X.max() - X.min())

y = loan_dataset['Personal Loan']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

knn_classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2, weights = 'distance')
knn_classifier.fit(X_train, y_train)

from sklearn.metrics import precision_score, recall_score, accuracy_score
print("Testing...\n")
y_pred_knn = knn_classifier.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_pred_knn))
print('Precision: ', precision_score(y_test, y_pred_knn))
print('Recall: ', recall_score(y_test, y_pred_knn))

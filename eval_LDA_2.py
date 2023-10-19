from LogisticRegression import LogisticRegression
from LDA import LDA

# importing the liabraries...
import numpy as np
import pandas as pd
import random
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions

from sklearn.datasets import load_iris
iris = load_iris()

data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
target = pd.DataFrame(data=iris['target'], columns=['target'])


from sklearn.metrics import accuracy_score
X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.1, random_state=42)


X_train_temp = X_train[['sepal length (cm)', 'sepal width (cm)']]
X_val_temp = X_val[['sepal length (cm)', 'sepal width (cm)']]

# Encode the target variable for binary classification
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# Object for Logistic regression class.
model = LDA()

# Call fit method for training...
model.fit(X_train_temp.to_numpy(), y_train_encoded)

# Call predict method for predicting...
y_pred = model.predict(X_val_temp.to_numpy())

# Print accuracy score
acc = accuracy_score(y_pred,y_val_encoded)

# Plot decision regions
plot_decision_regions(X_val_temp.to_numpy(), y_val_encoded, clf=model)
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.title('Decision regions for petal length and width')
plt.show()

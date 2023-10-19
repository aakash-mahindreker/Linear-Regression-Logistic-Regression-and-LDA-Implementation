from LinearRegression import LinearRegression
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


X = data[['sepal width (cm)','sepal length (cm)']]
y = data['petal width (cm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train, regularization=5, batch_size=10,epochs=2, learning_rate=0.01,val_split=0.1)

y_pred = model.predict(X_test)
mse = model.mean_squared_error(y_pred,y_test)
print("Mean Squared error = ",mse)
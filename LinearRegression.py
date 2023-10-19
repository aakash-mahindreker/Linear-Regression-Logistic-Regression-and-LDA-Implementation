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



import warnings
warnings.filterwarnings("ignore")


# creating the linear regression class...

class LinearRegression:

    def __init__(self):
        self.weights = None
        self.bias = None

    def calculate_loss(self, y_pred, y_val):
        loss = (np.square(y_pred - y_val)).mean()
        return loss

    def mean_squared_error(self, y_pred, y_test):
        y_test = y_test.to_numpy()
        n = y_test.shape[0]
        score = ((y_pred - y_test[:, None]) ** 2).mean()
        return score

    def random_permutation_indices(self, length):
        indices = list(range(length))
        random.shuffle(indices)
        return indices

    def dataframe_to_numpy(self, df):
        df = df.to_numpy()
        return df

    def split_train_validate(self, df, val_split):
        tl = 1 - val_split
        train_length = int(tl * len(df))
        train = df.iloc[:train_length]
        validate = df.iloc[train_length:]
        return train, validate

    def fit(self, X_train, y_train,
            epochs=3,
            learning_rate=0.01,
            regularization=0,
            batch_size=32,
            val_split=0.1):
        e = []
        l = []

        # initializing both weights, biases, lose
        self.weights = np.random.randn(X_train.shape[1], 1)
        self.bias = np.zeros((1, 1))
        self.previous_loss = float('inf')
        self.val_split = val_split
        patience = 5

        # splitting into batches...
        X_train, X_val = self.split_train_validate(X_train, val_split)
        y_train, y_val = self.split_train_validate(y_train, val_split)

        # converting the input data into numpy...
        X_train = self.dataframe_to_numpy(X_train)
        X_val = self.dataframe_to_numpy(X_val)
        y_train = self.dataframe_to_numpy(y_train)
        y_val = self.dataframe_to_numpy(y_val)

        # iterating through epochs...
        for i in range(epochs):
            e.append(i)
            # splitting bacthes...
            print(type(X_train))
            length = len(X_train)
            indices = self.random_permutation_indices(length)
            X_train = X_train[indices]
            y_train = y_train[indices]
            batch_number = math.ceil(len(X_train) / batch_size)

            for batch in range(batch_number):
                X_batch = X_train[i * batch_size: (i + 1) * batch_size]
                y_batch = y_train[i * batch_size: (i + 1) * batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias

                error = y_pred - y_batch  # self.mean_squared_error(y_pred,y_batch)

                weight_grads = 2 * np.dot(X_batch.T, error) / len(X_batch) + 2 * regularization * self.weights
                bias_grads = 2 * np.sum(error, axis=0) / len(X_batch)

                # Updating weights and biases....
                self.weights = self.weights - learning_rate * weight_grads
                self.bias = self.bias - learning_rate * bias_grads

            # doing prediction
            y_pred = np.dot(X_val, self.weights) + self.bias

            # calculating loss...
            loss = self.calculate_loss(y_pred, y_val)
            l.append(loss)
            # logic for early stopping...
            if loss > self.previous_loss:
                patience = patience - 1
                if patience == 0:
                    break
            else:
                patience = 5
                self.previous_loss = loss

        # plotting epochs vs loss...
        plt.plot(e, l)
        plt.xlabel("Epochs")
        plt.ylabel("losses")
        plt.show()

    def predict(self, X_test):
        X_test = X_test.to_numpy()
        preds = np.dot(X_test, self.weights) + self.bias
        return preds

# loading the iris data set..

from sklearn.datasets import load_iris
iris = load_iris()

data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
target = pd.DataFrame(data=iris['target'], columns=['target'])


mses = []
comparison = {"cases":[],"mse":[]}

X = data[['sepal width (cm)']]
y = data['petal width (cm)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train, regularization=5, batch_size=10,epochs=2, learning_rate=0.01,val_split=0.1)

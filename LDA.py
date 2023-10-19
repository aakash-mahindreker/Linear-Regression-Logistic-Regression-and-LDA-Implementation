import numpy as np
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

class LDA:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.mean_overall = None
        self.Sw = None
        self.Sb = None
        self.weights = None
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.mean_overall = np.mean(X, axis=0)
        class_means = np.array([np.mean(X[y == i], axis=0) for i in np.unique(y)])
        self.Sw = self._calculate_within_class_scatter_matrix(X, y, class_means)
        self.Sb = self._calculate_between_class_scatter_matrix(X, y, class_means)
        self.weights = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            y_pred = self._predict(X)
            errors = y - y_pred
            gradient = -2 * X.T.dot(errors) / X.shape[0]
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        y_pred = np.sign(self._predict(X))
        return y_pred

    def _predict(self, X):
        return X.dot(self.weights)

    def _calculate_within_class_scatter_matrix(self, X, y, class_means):
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for i in np.unique(y):
            class_samples = X[y == i]
            class_mean = class_means[i]
            Sw += (class_samples - class_mean).T.dot(class_samples - class_mean)
        return Sw

    def _calculate_between_class_scatter_matrix(self, X, y, class_means):
        Sb = np.zeros((X.shape[1], X.shape[1]))
        for i in np.unique(y):
            class_samples = X[y == i]
            class_mean = class_means[i]
            Sb += np.outer(class_mean - self.mean_overall, class_mean - self.mean_overall) * class_samples.shape[0]
        return Sb

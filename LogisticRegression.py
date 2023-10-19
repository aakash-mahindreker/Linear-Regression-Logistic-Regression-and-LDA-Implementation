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
class LogisticRegression:

    def __init__(self, lr=0.01, epochs=10):
        self.lr = lr
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        X = self.add_bias(X)
        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights -= self.lr * gradient
            cost = self.calculate_cost(h, y)
            self.losses.append(cost)

    def predict(self, X, threshold=0.5):
        X = self.add_bias(X)
        z = np.dot(X, self.weights)
        y_pred_prob = self.sigmoid(z)

        if threshold is None or threshold == 0:
            return y_pred_prob

        y_pred = np.where(y_pred_prob >= threshold, 1, 0)
        return y_pred

    def accuracy_score(self, y_pred, y_true, threshold=0.5):
        accuracy = np.mean(y_pred == y_true)
        return accuracy

    def recall_score(self, X, y_true, threshold=0.5):
        y_pred = self.predict(X, threshold)
        true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
        false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
        recall = true_positives / (true_positives + false_negatives)
        return recall

    def f1_score(self, X, y_true, threshold=0.5):
        precision = self.precision_score(X, y_true, threshold)
        recall = self.recall_score(X, y_true, threshold)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def calculate_cost(self, h, y):
        epsilon = 1e-5  # small value to avoid division by zero
        cost = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost
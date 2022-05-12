'''

Linear regression from scratch

Regression we want to predict continuous valuies
Classication we want to predict discrete function

Aproximate the underlying trend with a linear function and predict the values - linear regression

eg: ypred = wx + b

Loss function is usually a function defined on a data point, prediction and label, and measures the penalty. 

Cost function is usually more general. It might be a sum of loss functions over your training set plus some model complexity penalty (regularization).

Objective function is the most general term for any function that you optimize during training


cost_function use is MSE which is the sum of the diff between the 
y and y pred squwared for each data sample x and div by the number of total
samples in the dataset.

gradient decent is an iterative process to find the minimum of the cost function.


'''

import numpy as np
from torch import binary_cross_entropy_with_logits

class LinearRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.weights = np.zeros(n_features)
        
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias 

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw   
            self.bias -= self.lr * db
        

    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

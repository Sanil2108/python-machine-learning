# THE ONE THAT WORKS PERFECTLY

from __future__ import division
import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def hypothesis(theta, X):
    # X is a mx(n+1) matrix
    # theta is a (n+1)x1 matrix

    # Both are 2 dimensional arrays and not
    # np.matrix
    X_np = np.matrix(X)
    theta_np = np.matrix(theta)
    return np.array(sigmoid(X_np * theta_np))[0][0]

def cost(theta, X, y):
    cost = 0
    m = len(y)
    for i in range(m):
        cost = cost+(y[i][0] * np.log(hypothesis(theta, X[i])) + (1-y[i][0])*np.log(1-hypothesis(theta, X[i])))
    return -cost/m


def gradient(X, y, alpha):
    m = len(X)
    theta = [[0] for i in range(len(X[0]))]
    iter_count = 1000
    for i2 in range(iter_count):
        for i in range(m):
            for j in range(len(X[0])):
                theta[j] = theta[j] - alpha * (hypothesis(theta, X[i]) - y[i]) * X[i][j] / m

    # print(type(theta))
    # print(theta)

    return theta

def predict(y):
    # new_y=[0] * len(y)
    # for i in range(len(y)):
    #     if y[i]>=0.5:
    #         new_y[i] = 1
    # return new_y

    if y>=0.5:
        return 1
    else:
        return 0

# This is the 'OR' function
X = [
    [1, 0, 0],
    [1, 1, 0],
    [1, 0, 1],
    [1, 1, 1]
]
y = [
    [0],
    [1],
    [1],
    [1]
]


print(predict(hypothesis(gradient(X, y, 0.1), [[1, 0, 1]])))
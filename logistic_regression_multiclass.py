# THE ONE THAT WORKS PERFECTLY

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

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


def gradient(X, y, alpha=0.01):
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

def multiclass(X, y, new_x, all=[0, 1, 2]):
    p=3
    max_hypothesis_val=-1
    max_hypothesis=-1
    combined_arr_x, combined_arr_y = get_combined_arr(X, y, p)

    for i in range(p):
        rest = get_except(all, i)
        final_temp=[]
        final_temp_y = []
        for j in range(len(combined_arr_y)):
            if combined_arr_y[j][0][0] in rest:
                final_temp=combine(final_temp, combined_arr_x[j])
                final_temp_y=combine(final_temp_y, combined_arr_y[j])
        #final temp set to the value to use as X in logistic regression
        #y value to be used is combined_arr_y[i]
        print(final_temp)

        final_temp_y=combine(final_temp_y, combined_arr_y[i])
        final_temp=combine(final_temp, combined_arr_x[i])

        for j in range(len(final_temp_y)):
            if final_temp_y[j][0] != i:
                final_temp_y[j] = [0]
            else:
                final_temp_y[j] = [1]

        theta_temp = gradient(final_temp, final_temp_y)
        print(theta_temp)
        hypothesis_temp = hypothesis(theta_temp, new_x)

        if hypothesis_temp>max_hypothesis:
            max_hypothesis = hypothesis_temp
            max_hypothesis_val = i
            print(max_hypothesis_val)

        # plot(combined_arr_x)
        # plot_line(theta_temp, -10, 10)


    return max_hypothesis_val

def get_except(old_arr, exception):
    new_arr=[0 for i in range(len(old_arr) - 1)]
    j=0
    for i in range(len(old_arr)):
        if old_arr[i] != exception:
            new_arr[j] = old_arr[i]
            j += 1
    return new_arr

def get_combined_arr(X, y, p):
    combined_arr_x = []
    combined_arr_y = []
    for i in range(p):
        temp_combined_arr_x = []
        temp_combined_arr_y = []
        for j in range(len(y)):
            if y[j][0] == i:
                temp_combined_arr_x.append(X[j])
                temp_combined_arr_y.append(y[j])
        combined_arr_x.append(temp_combined_arr_x)
        combined_arr_y.append(temp_combined_arr_y)

    return combined_arr_x, combined_arr_y


X = [
    [1, 1, 10],
    [1, 2, 12],
    [1, 3, 9],
    [1, 3, 13],
    [1, 11, 1],
    [1, 22, 2],
    [1, 13, 4],
    [1, 15, 3],
    [1, 1, 0],
    [1, 2, 1],
    [1, 3, 4],
    [1, 3, 1]
]
y = [
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [2],
    [2],
    [2],
    [2]
]

def combine(s, t):
    for i in range(len(t)):
        s.append(t[i])
    return s

def plot(X):
    for i in range(len(X[0])):
        plt.plot(X[0][i][1], X[0][i][2], 'o')
    for i in range(len(X[1])):
        plt.plot(X[0][i][1], X[0][i][2], 'r')
    for i in range(len(X[2])):
        plt.plot(X[0][i][1], X[0][i][2], 'x')

def plot_line(theta, start, end):
    y=[]
    x=[]
    for i in range(start, int((end-start)/2), end):
        x.append(i)
        y.append((theta[0]+theta[1]*i)/theta[2])

    plt.plot(x, y)



print(multiclass(X, y, [1, 10, 1]))
#plt.show()
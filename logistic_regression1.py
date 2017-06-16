import numpy as np
import matplotlib.pyplot as plt

all_cost=[]

def logistic(z):
    return 1/(1+np.exp(-z))

def hypothesis(theta, X):
    return logistic(np.array(np.matrix(X)*np.transpose(np.matrix(theta))))[0][0]
#     return getY(theta, X)

def cost(theta, X, y):
    m=len(y)
    total=0
    for i in range(m):
        total+=(y[i]*np.log(hypothesis(theta, X[i])) + (1-y[i])*np.log(1-hypothesis(theta, X[i])))
    return -total/m

def gradient_descent(X, y, alpha):
    tempCost=1000
    while(tempCost>0.01):
        for j in range(len(theta)):
            pd=0
            for i in range(len(y)):
                pd+=(hypothesis(theta, X[i])-y[i])*X[i][j]
            theta[j]=theta[j]-alpha*pd
        all_cost.append(tempCost)
        if(tempCost-cost(theta, X, y)<1e-50):
            break
        tempCost=cost(theta, X, y)
        print(tempCost)
    print(theta)
    # temp_x = np.linspace(0, len(all_cost), len(all_cost) + 1)
    # for i in range(len(all_cost)):
    #     plt.plot(temp_x[i], all_cost[i], 'ro')
    # plt.show()
    return theta

#X is an (n+1) row vector
def getY(theta, X):
    if(np.array(np.matrix(X)*np.transpose(np.matrix(theta)))>=0.5):
        return 1
    else:
        return 0


# new dataset for a circular decision boundary
X = [
    [1, 0, 0, 0, 0, 0],
    [1, 0.5, 0.25, -0.5, 0.25, -0.25],
    [1, 0.5, 0.25, 0.5, 0.25, 0.25],
    [1, - 0.5, 0.25, -0.5, 0.25, 0.25],
    [1, -0.5, 0.25, 0.5, 0.25, -0.25],

    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, -1, 1, -1],
    [1, -1, 1, 1, 1, -1],
    [1, -1, 1, -1, 1, 1],
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, -1, 1, 0],
    [1, 1, 1, 0, 0, 0],
    [1, -1, 1, 0, 0, 0]
]

y = [
    0,
    0,
    0,
    0,
    0,

    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
]

theta = [
    0,
    0,
    0,
    0,
    0,
    0
]

alpha = 0.05

gradient_descent(X, y, alpha)
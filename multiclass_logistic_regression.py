#multiclass logisitic regression
import numpy as np
import matplotlib.pyplot as plt

all_cost=[]

def logistic(z):
    return 1/(1+np.exp(-z))

def hypothesis(theta, X):
    return logistic(np.array(np.matrix(X)*np.transpose(np.matrix(theta))))[0][0]
#     return getY(theta, X)

def cost(theta, X, y, y_val):
    m=len(y)
    total=0
    for i in range(m):
        temp_y=y[i]
        if(y[i]==y_val):
            temp_y=1
        else:
            temp_y=0
        total+=(temp_y*np.log(hypothesis(theta, X[i])) + (1-temp_y)*np.log(1-hypothesis(theta, X[i])))
    return -total/m

#y_val denotes the current class being tested
def gradient_descent(X, y, alpha, y_val):
    theta=[0]*len(X[0])
    tempCost=1000
    while(tempCost>0.01):
        for j in range(len(theta)):
            pd=0
            for i in range(len(y)):
                temp_y=y[i]
                if(y[i]==y_val):
                    temp_y=1
                else:
                    temp_y=0
                pd+=(hypothesis(theta, X[i])-temp_y)*X[i][j]
            theta[j]=theta[j]-alpha*pd
        all_cost.append(tempCost)
        if(tempCost-cost(theta, X, y, y_val)<1e-5):
            break
        tempCost=cost(theta, X, y, y_val)
#         print(tempCost)
#     print(theta)
    # temp_x = np.linspace(0, len(all_cost), len(all_cost) + 1)
    # for i in range(len(all_cost)):
    #     plt.plot(temp_x[i], all_cost[i], 'ro')
    # plt.show()
    return theta

#X is an (n+1) row vector
def getY(theta, X, no_of_classes):
    max_=0
    max_hypothesis=-1
    for i in range(no_of_classes):
        temp_hypothesis=np.array(np.matrix(X)*np.transpose(np.matrix(theta[i])))[0][0]
        if(temp_hypothesis>max_hypothesis):
            hypothesis=i
    return hypothesis


# class count starts from 0
def multiclass(X, y, alpha, classes):
    all_thetas = []
    for i in range(classes):
        all_thetas.append(gradient_descent(X, y, alpha, i))

    return all_thetas

#dataset 1
X=[
    [1, 0, 2],
    [1, 1, 4],
    [1, 2, 6],
    [1, 3, 8],
    [1, 4, 10],
    [1, 5, 12]
]

y=[
    0,
    0,
    1,
    1,
    2,
    2
]

# dataset 2
X = [
    [1, 3, 3],
    [1, 3, 4],
    [1, 3, 5],
    [1, 4, 3],
    [1, 4, 4],
    [1, 4, 5],
    [1, 5, 3],
    [1, 5, 4],
    [1, 5, 5],

    [1, 3, 3],
    [1, 3, 2],
    [1, 3, 1],
    [1, 2, 3],
    [1, 2, 2],
    [1, 2, 1],
    [1, 1, 3],
    [1, 1, 2],
    [1, 1, 1],

    [1, 4, 0],
    [1, 4, 1],
    [1, 4, 2],
    [1, 5, 0],
    [1, 5, 1],
    [1, 5, 2],
    [1, 6, 0],
    [1, 6, 1],
    [1, 6, 2]
]

y = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2
]

theta=multiclass(X, y, 0.01, 3)
print(theta)

getY(theta, [1,3.5,3.5], 3)
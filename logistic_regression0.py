import numpy as np
import matplotlib.pyplot as plt


# for multiclass. maybe i can change  1/(1+np.exp(-np.array(z)[0][0])) to
# n/(1+np.exp(-np.array(z)[0][0])), where n is the max value of y or the
# number of classes -1
def hypothesis(theta, X):
    z=np.matrix(X)*np.transpose(np.matrix(theta))
    return 3/(1+np.exp(-np.array(z)[0][0]))

def cost_unvectorized(X, Y, theta):
    total=0
    m=len(Y)
    for i in range(m):
        total=total+Y[i]*np.log(hypothesis(theta, X[i]))+(1-Y[i])*np.log(1-hypothesis(theta, X[i]))
    total=-1*total/m
    return total

def cost_vectorized(X, Y, theta):
    cost=-np.transpose(Y)*np.log(hypothesis(theta, X))-np.transpose(1-np.matrix(Y))*np.log(1-hypothesis(theta, X))
    return (1/len(Y))*cost

def gradient_descent(X, Y, alpha):
    n=len(X[0])
    m=len(Y)
    theta=[0] * n
    i=1000
    while i>0:
        for j in range(n):
            pd=0
            for k in range(m):
                pd=pd+(hypothesis(theta, X[k])-Y[k])*X[k][j]
            theta[j]=theta[j]-(alpha/m)*pd
            i-=1

    print(theta)
    return theta


def plot_graph(theta, n=100):
    x=np.linspace(-n, n, 20*n)
    y=[0 for i in range(2*n)]
    for i in range(len(x)):
        y[i]=1/(1+np.exp(-np.transpose(theta)*x[i]))
    plt.plot(x, y)


Y=[0, 0, 1, 1, 2, 2]
X=[[1, 0, 0], [1, 1, 1], [1, 5, 1], [1, 5, 2], [1, 1, 7], [1, 2, 7]]
theta=[1, 0, 0]
# print(cost_unvectorized(X, Y, theta))
theta = gradient_descent(X, Y, 0.1)
print(hypothesis(theta, [1, 1, 15]))
# plot_graph(theta)
# plt.axis([-3, 3, 0, 2])
# plt.show()
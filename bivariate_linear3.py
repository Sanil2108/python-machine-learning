import matplotlib.pyplot as plt
import random

    #turning this into bivariate, theta0 as of this cannot be set zero at runtime
#removing all default values of theta0 and making it a permanent formal parameter
def plot_(arr, n=100):
    y = [0] * n
    for i in range(n):
        for j in range(len(arr)):
            y[i] += arr[j] * pow(i, j)
    plt.plot(y)


def cost_fn(m, x, y, theta):
    theta1=theta[1]
    theta0=theta[0]
    cost = 0
    costs = [0]*m
    for i in range(m):
        costs[i]=cost
        cost += pow((hypothesis_fn(theta, x[i]) - y[i]), 2)
    return (1 / 2 * m) * cost


def hypothesis_fn(theta, x):
    return theta[1] * x + theta[0]

#for X and Y ia m using f(x) = 2*x + 1
# X = [i for i in range(10)]
# Y = [2 * X[i] + 1 for i in range(len(X))]

X = [5, 2]
Y = [1, 6]
m = len(X)
alpha = 0.1  # Learning rate

theta1 = 0.5

#theta0 randomly computated
theta0 = 0.5

n = 10
old_pd=100
old_cost=100
n_default = 10000

costs = []

#for use in loops and decision making statements,
#and also setting the basis for multivariate linear and polynomial,
#theta1 and theta0 are being converted to a list which contains both
theta = [theta0, theta1]
#random variable
random_var = 1
random_bool = True

while(random_bool):
    random_var = 1
    while (random_var>=0):

        # calculate partial derivative
        pd = 0
        for i in range(m):
            pd += hypothesis_fn(theta, X[i]) - Y[i]
        pd /= m
        #
        # n -= 1


        theta[random_var]=theta[random_var]-alpha*pd
        current_cost = cost_fn(m, X, Y, theta)
        costs.append(cost_fn(m, X, Y, theta))

        if(cost_fn(m, X, Y, theta) <= 2 or abs(old_cost-current_cost)<1e-100 or n<=1):
            random_bool=False

        print(current_cost, old_cost-current_cost, abs(old_cost-current_cost), end='\n')

        old_cost=current_cost

        n-=1

        random_var -= 1

    n = n_default


print(theta, 'cost = ', cost_fn(m, X, Y, theta))

#plotting the graph

plt.subplot(221)

plot_([theta[0], theta[1]])
plt.plot(X, Y, 'ro')
plt.axis([0, 1.2*X[-1], 0, 1.2*Y[-1]])

plt.subplot(222)
plt.plot(costs)

print('\n\n\n', theta[1] * 3 + theta[0])
plt.show()
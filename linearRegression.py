def hypothesis(theta, x):
    return theta[0]+theta[1]*x

def cost(theta, x, y):
    m=len(theta)
    total=0
    for i in range(m):
        total+=(hypothesis(theta, x[i])-y[i])*(hypothesis(theta, x[i])-y[i])
    return total/(2*m)

all_theta1s = []
all_theta0s = []
all_costs=[]

def gradient_descent(x, y, alpha):
    theta = [0, 0]
    k = 0
    m = len(x)
    while (k < 100):
        pd = 0
        for i in range(len(x)):
            pd += hypothesis(theta, x[i]) - y[i]
        pd = pd * (alpha / m)
        theta[0] = theta[0] - pd
        theta[1] = theta[1] - pd * x[i]

        all_theta1s.append(theta[1])
        all_theta0s.append(theta[0])

        cost_current=cost(theta, x, y)
        all_costs.append(cost_current)

        print(theta[1], ' ', theta[0], ' cost -  ', cost_current)
        k += 1


x=[1, 2, 4, 7]
y=[9, 17, 39, 64]
alpha=0.01

gradient_descent(x, y, alpha)

import matplotlib.pyplot as plt
x_values=[i for i in range(len(all_theta0s))]
plt.subplot(2, 2, 1)
plt.plot(x_values, all_theta1s)
plt.title('Theta 1')
plt.subplot(2, 2, 2)
plt.plot(x_values, all_theta0s)
plt.title('Theta 0')
plt.subplot(2, 2, 3)
plt.plot(x_values, all_costs)
plt.title('Cost')

plt.show();
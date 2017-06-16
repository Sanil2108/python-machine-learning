import matplotlib.pyplot as plt
import numpy as np

def hypothesis(theta,  X):
    X=np.matrix(X)
    theta=np.transpose(np.matrix(theta))
    return np.array(X*theta)[0][0]

def cost(theta, X, y):
    total=0
    for i in range(len(y)):
        total=total+(hypothesis(theta, X[i])-y[i])**2
    return total/(2*len(y))

all_thetas=[]
all_costs=[]

def gradient_descent(X, y, alpha):
    m=len(y)
    n=len(X[0])-1
    theta=[0 for i in range(n+1)]
    k=0
    cost_too_low=False
    while(k<100 and cost_too_low==False):
        #thetaTemp is used for simultaneous update
        thetaTemp=[theta[i] for i in range(len(theta))]
        prev_cost=0
        for j in range(n+1):
            pd=0
            for i in range(n+1):
                pd+=(hypothesis(thetaTemp, X[i])-y[i])*X[i][j]
            theta[j]=theta[j]-alpha*pd
            temp_cost=cost(theta, X, y)
            # print(temp_cost)
            all_costs.append(temp_cost)
            all_thetas.append(thetaTemp)
            print(temp_cost-prev_cost)
            print(temp_cost)
            if(temp_cost-prev_cost<1 and k>5):
                cost_too_low=True
            k+=1
            prev_cost=temp_cost

    return theta

X=[
    [1, 2, 4],
    [1, 4, 5],
    [1, 6, 6],
    [1, 8, 7],
    [1, 10, 8],
    [1, 12, 9]
]

y=[
    17, 24, 31, 38, 45, 52
]

# thetaTemp=[
#     1, 2, 3
# ]

# print(hypothesis(thetaTemp, X[0]))
# print(cost(thetaTemp, X, y))
print(gradient_descent(X, y, 0.01))

# print(all_thetas)

#plotting cost and theta
x=np.linspace(0, len(all_thetas), len(all_thetas))

plt.subplot(2, 2, 1)
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.plot(x, all_costs)
plt.title('Cost')

plt.subplot(2, 2, 2)
plt.xlabel('Iterations')
plt.ylabel('Value')
all_thetas_matrix=np.transpose(np.matrix(all_thetas))

plt.plot(np.linspace(1, len(all_thetas[0]), len(all_thetas[0])), all_thetas_matrix)
plt.title('Thetas')

plt.show()
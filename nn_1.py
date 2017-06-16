import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.structure = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def feedforward(self, X):
        all_z = []
        all_a = [np.matrix(X)]
        a = X

        for i in range(self.num_layers-1):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, a) + b
            all_z.append(z)
            a = self.sigmoid(z)
            all_a.append(np.matrix(a))

        return all_a[-1], all_z[-1]

    def gradient_cost(self, a, y):
        return a-y

    def gradient_descent(self, X, y, eta):
        #X contains all training examples
        #y contains all results
        tempCost=1000
        m=len(y)
        while(abs(tempCost)>1e-4):
            tempCost=0
            for i in range(len(X)):
                a, z=self.feedforward(X[i])
                cost=self.gradient_cost(a, y[i])
                sp=self.sigmoid_prime(z)
                gamma = np.multiply(cost, sp)
                tempCost+=np.array(gamma)[0][0]
                # print('Error = ', gamma)
                nabla_b, nabla_w = self.backpropagate(X[i], y[i])

                # for i in range(len(self.weights)):
                for i in range(len(self.weights)):
                    for j in range(len(self.weights[i])):
                        self.weights[i][j]=self.weights[i][j]-(eta*nabla_w[i][j])/m

                for i in range(len(self.biases)):
                    for j in range(len(self.biases[i])):
                        self.biases[i][j]=self.biases[i][j]-(eta*nabla_b[i][j])/m

            tempCost/=m
            print('Cost - ', abs(tempCost))

        return self.weights, self.biases

    def backpropagate(self, X, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation=X
        activations=[X]
        zs=[]

        for w, b in zip(self.weights, self.biases):
            z=np.dot(w, activation)+b
            zs.append(z)
            activation=self.sigmoid(z)
            activations.append(activation)

        delta = np.multiply(self.gradient_cost(activations[-1], y), self.sigmoid_prime(zs[-1]))
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta, np.transpose(activations[-2]))

        for i in range(2, self.num_layers):
            sp=self.sigmoid_prime(zs[-i])
            w=self.weights[-i+1]
            temp=np.dot(np.transpose(np.matrix(w)), delta)
            #delta is scherr product sp. not dot product sp
            delta=np.multiply(temp, sp)

            nabla_b[-i]=delta
            nabla_w[-i]=np.dot(delta, np.transpose(np.matrix(activations[-i-1])))

        return nabla_b, nabla_w


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feedforward(X, weights, biases, num_layers):
    all_z = []
    all_a = [np.matrix(X)]
    a = X

    for i in range(num_layers - 1):
        w = weights[i]
        b = biases[i]
        z = np.dot(w, a)
        z = z + b
        all_z.append(z)
        a = sigmoid(z)
        all_a.append(np.matrix(a))

    return all_a[-1]

n=Network([2,1])
X=[
    [
        [0],[0]
    ],[
        [0],[1]
    ],[
        [1],[0]
    ],[
        [1],[1]
    ]
]
y=[
    [[0]],
    [[1]],
    [[1]],
    [[1]]
]

weights, biases = n.gradient_descent(X, y, 1)
print(feedforward([[0],[0]], weights, biases, 2))
# libraries
import numpy as np
from sklearn.utils import shuffle
import random

class Network:
    def __init__(self, layer_dims, X, Y):
        """
        :param layer_dims:
        :param X: it is the collection of labelled inputs arranged in order (m,n)
        m - no. of independent variable in one sample, n - no. of samples
        :param Y: output set with order (1,n)
        """
        self.layer_dims = layer_dims
        self.params = dict()
        self.X = X
        self.Y = Y
        self.X_train = X[:, 0:int(0.75 * X.shape[1])]
        self.Y_train = Y[:, 0:int(0.75 * Y.shape[1])]
        self.X_test = X[:, int(0.75 * X.shape[1]):]
        self.Y_test = Y[:, int(0.75 * X.shape[1]):]

    def initialize_parameters(self):
        for i in range(1, len(self.layer_dims)):
            self.params["W" + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.01
            self.params["b" + str(i)] = np.zeros((self.layer_dims[i], 1))

    def start(self, num_iterations=200, print_cost=False, learning_rate=0.09):
        m = self.Y_test.shape[1]
        for i in range(num_iterations):
            self.X_train, self.Y_train = shuffle(self.X_train.T, self.Y_train.T, random_state=random.randint(0,i))
            self.X_train = self.X_train.T
            self.Y_train = self.Y_train.T

            AL, caches = self.forward_propagation(self.X_train)

            cost = self.compute_cost(AL)
            grads = self.backward_propagation(AL, caches)

            self.update_parameters(grads, learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        probabilities = self.predict(self.X_test)
        print("Accuracy: " + str(np.sum((probabilities == self.Y_test) / probabilities.shape[1])))

    def forward_propagation(self, X):
        A = X
        caches = []
        L = len(self.params) // 2

        for i in range(1, L):
            A_prev = A
            Wl = self.params["W" + str(i)]
            Z = np.dot(Wl, A_prev) + self.params["b" + str(i)]
            linear_cache = (A_prev, self.params["W" + str(i)], self.params["b" + str(i)])
            A, activation_cache = relu(Z)
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        ZL = np.dot(self.params["W" + str(L)], A) + self.params["b" + str(L)]
        linear_cache = (A, self.params["W" + str(L)], self.params["b" + str(L)])
        AL, activation_cache = sigmoid(ZL)
        cache = (linear_cache, activation_cache)
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL):
        """
        returns cost computed
        :rtype: decimal
        """
        m = self.Y_train.shape[1]
        cost = -1 / m * np.sum(self.Y_train * np.log(AL) + (1 - self.Y_train) * np.log(1 - AL))
        cost = np.squeeze(cost)
        return cost

    def backward_propagation(self, AL, caches):
        """
        gives a dict of differentiated parameters
        :rtype: dict
        """
        grads = dict()
        L = len(caches)
        m = AL.shape[1]
        Y = self.Y_train
        # derivative of a sigmoid function
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[-1]

        dZ = diff_sigmoid(dAL, current_cache[1])
        A_prev, W, b = current_cache[0]

        grads['dW' + str(L)] = 1 / m * np.dot(dZ, A_prev.T)
        grads['db' + str(L)] = 1 / m * np.sum(dZ, keepdims=True, axis=1)
        grads['dA' + str(L)] = np.dot(W.T, dZ)
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev = grads['dA' + str(l + 2)]

            dZ = diff_relu(dA_prev, current_cache[1])
            # current_cache[1] = value of Z at that layer
            A_prev, W, b = current_cache[0]

            grads['dW' + str(l+1)] = 1 / m * np.dot(dZ, A_prev.T)
            grads['db' + str(l+1)] = 1 / m * np.sum(dZ, keepdims=True, axis=1)
            grads['dA' + str(l+1)] = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, len(self.params)//2 + 1):
            self.params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
            self.params["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    def predict(self, X):

        p, caches = self.forward_propagation(X)
        for i in range(p.shape[1]):
            # print(p[0, i])
            if p[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        return p


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z)), Z


def relu(Z):
    return np.maximum(0.0, Z), Z


def diff_sigmoid(dA, activation_cache):
    Z = activation_cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def diff_relu(dA, activation_cache):
    Z = activation_cache
    # converting dZ to a correct object
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

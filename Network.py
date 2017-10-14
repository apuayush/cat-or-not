# libraries
import numpy as np


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
        self.Y_train = Y[:, int(0.75 * X.shape[1]):]

    def initialize_parameters(self):
        for i in range(1, len(self.layer_dims)):
            self.params["W" + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1]) * 0.01
            self.params["b" + str(i)] = np.zeros((self.layer_dims[i], 1))

    def start(self, num_iterations=200, print_cost=False, learning_rate=3000):
        for i in range(num_iterations):
            AL, caches = self.forward_propagation()

            cost = self.compute_cost(AL)
            grads = self.backward_propagation(AL, caches)

            self.update_parameters(grads, learning_rate)

            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

    def forward_propagation(self):
        A = self.X_train
        caches = []
        L = len(self.params) // 2
        for i in range(1, L):
            A_prev = A
            Z = np.dot(self.params["W" + str(i)], A_prev) + self.params["b" + str(i)]
            linear_cache = (A_prev, self.params["W" + str(i)], self.params["b" + str(i)])
            A, activation_cache = relu(Z)
            cache = (linear_cache, activation_cache)
            caches.append(cache)

        ZL = np.dot(self.params["W" + str(L)], A) + self.params["b" + str(L)]
        linear_cache = (A, self.params["W" + str(L)], self.params["b" + str(L)])
        AL, activation_cache = sigmoid(Z)
        cache = (linear_cache, activation_cache)
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL):
        """
        returns cost computed
        :rtype: decimal
        """
        pass

    def backward_propagation(self, AL, caches):
        """
        gives a dict of differentiated parameters
        :rtype: dict
        """
        pass

    def update_parameters(self):
        pass


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def relu(Z):
    return max(0.0, Z)

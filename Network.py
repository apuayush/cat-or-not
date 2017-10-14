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
        self.caches = list()
        self.X = X
        self.Y = Y
        self.X_train = X[:, 0:int(0.75*X.shape[1])]
        self.Y_train = Y[:, 0:int(0.75*Y.shape[1])]
        self.X_test = X[:, int(0.75*X.shape[1]):]
        self.Y_train = Y[:, int(0.75*X.shape[1]):]

    def initialize_parameters(self):
        for i in range(1, len(self.layer_dims)):
            self.params["W" + str(i)] = np.random.randn(self.layer_dims[i], self.layer_dims[i - 1])*0.01
            self.params["b" + str(i)] = np.zeros((self.layer_dims[i], 1))

    def forward_propagation(self):
        A = self.X
        for i in range(1, len(self.layer_dims)-1):
            A, cache = self.linear_forward(i, 'relu', A)

    def linear_forward(self , layer_no, activation_func, A_prev):
        Z = np.dot(self.params['W'+str(layer_no)], A_prev) +  self.params["b" + str(layer_no)]
        if activation_func == 'sigmoid':
            A = sigmoid(Z)

        else:
            A = relu(Z)
        cache = (Z)
        return A, cache






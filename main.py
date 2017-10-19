# module libraries
from image_classifier import image_classifier
from Network import Network
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
# other libs
import numpy as np
import sys


def pre_processing(Y):
    Y = Y.reshape(1, Y.shape[0])
    onehot = OneHotEncoder(categorical_features=[1])
    Y = onehot.fit_transform(Y).toarray()
    return Y

path = "/home/apurvnit/Projects/cat-or-not/data2"
if len(sys.argv) == 2:
    path = sys.argv[1]

X, Y = image_classifier(path)
print(X.shape)
Y = pre_processing(Y)

X, Y = shuffle(X.T, Y.T, random_state=2)
X = X.T
Y = Y.T

net = Network([12288, 100, 62, 22, 16, 3, 1], X, Y)
net.initialize_parameters()
net.start(num_iterations=2, print_cost=True, learning_rate=0.0075)

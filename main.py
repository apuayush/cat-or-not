# module libraries
from image_classifier import image_classifier
from Network import Network
from sklearn.preprocessing import OneHotEncoder
# other libs


def pre_processing(X, Y):
    Y = Y.reshape(1, Y.shape[0])
    onehot = OneHotEncoder(categorical_features=[1])
    Y = onehot.fit_transform(Y).toarray()
    X = X.reshape(X.shape[0], -1).T / 255
    return X, Y


X, Y = image_classifier()
X, Y = pre_processing(X, Y)

net = Network([12288, 50, 25, 12, 6, 3, 1], X, Y)
net.initialize_parameters()
net.start(num_iterations=1, print_cost=True)

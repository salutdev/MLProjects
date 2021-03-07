from calculations.utils import Utils
from matplotlib.pyplot import axis
from ..utils import Utils
import numpy as np

class NNGradientDescent:

    def calculate_params(self, x1, x2, y):

        X = np.row_stack((x1, x2))
        Y = np.reshape(y, (1, -1))

        m = X.shape[1]

        mu = X.sum(axis=1, keepdims=True)/m
        X = X - mu

        X_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / X_norm

        learning_rate = 1
        num_iterations = 5000
        n_h = 4

        parameters = self.nn_model(X, Y, n_h, num_iterations, learning_rate)

        return parameters, mu, X_norm

    def nn_model(self, X, Y, n_h, num_iterations, learning_rate):

        np.random.seed(3)

        n_x, n_h, n_y = self.layer_sizes(X, Y)

        parameters = self.initialize_parameters(n_x, n_h, n_y)

        for i in range(0, num_iterations):

            A2, cache = NNGradientDescent.forward_propagation(X, parameters)
            cost = self.compute_cost(A2, Y)
            grads = self.backward_propagation(parameters, cache, X, Y)
            parameters = self.update_parameters(parameters, grads, learning_rate)

            if i % 1000 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
        
        return parameters

    @staticmethod
    def forward_propagation(X, parameters):

        W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = Utils.sigmoid(Z2)

        assert(A2.shape == (1, X.shape[1]))
        cache = {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
            }

        return A2, cache

    def backward_propagation(self, parameters, cache, X, Y):

        m = X.shape[1]

        W1 = parameters["W1"]
        W2 = parameters["W2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
        dW2 = 1/m * np.dot(dZ2, A1.T)
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1-np.power(A1, 2)))
        dW1 = 1/m * np.dot(dZ1, X.T)
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
            }

        return grads

    def update_parameters(self, parameters, grads, learning_rate):

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
            }

        return parameters

    def compute_cost(self, A2, Y):

        m = Y.shape[1]

        logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
        cost = -np.sum(logprobs)/m

        cost = float(np.squeeze(cost))
        assert(isinstance(cost, float))

        return cost

    def layer_sizes(self, X, Y):
        n_x = X.shape[0]
        n_h = 4
        n_y = Y.shape[0]
        return (n_x, n_h, n_y)

    def initialize_parameters(self, n_x, n_h, n_y):
        np.random.seed(2)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {
            "W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2
            }

        return parameters
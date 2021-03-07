from calculations.neural_network.nn_gradient_descent import NNGradientDescent
import matplotlib.pyplot as plt
import numpy as np
import time

class NNGraphics:

    @staticmethod
    def plot_lines():
        x1, y1 = [0, 12, 24], [1, 4, 31]
        x2, y2 = [1, 10], [3, 2]
        plt.plot(x1, y1, marker = 'o')
        plt.show()

    @staticmethod
    def plot_dots(x1, x2, Y, parameters, mu, X_norm):

        low_x1 = 0
        hi_x1 = 9000

        low_x2 = 0
        hi_x2 = 100

        # Test desicion boundary
        #NNGraphics.show_test_dot(parameters, mu, X_norm)

        # initial boundary line
        lx1, lx2 = [low_x1, hi_x1], [31, 70]
        plt.plot(lx1, lx2, marker = 'o')

        # dots
        val = np.array(Y)
        colors = np.where(val == 1, 'g', 'r')
        plt.scatter(x1, x2, c = colors, marker='o', alpha=1)

        plt.axis([low_x1, hi_x1, low_x2, hi_x2])
        plt.xlabel('Parameter X1')
        plt.ylabel('Parameter X2')
       
        plt.show()

    @staticmethod
    def plot_dots2(x1, x2, Y, parameters, mu, X_norm):

        low_x1 = 0
        hi_x1 = 9000

        low_x2 = 0
        hi_x2 = 100

        # Test desicion boundary
        NNGraphics.show_test_dot(parameters, mu, X_norm)

        # initial boundary lines
        lx1, lx2 = [low_x1, hi_x1], [31, 70]
        plt.plot(lx1, lx2, marker = 'o', color='b')

        lx1, lx2 = [3000, 7000], [100, 0]
        plt.plot(lx1, lx2, marker = 'o', color='b')

        # dots
        val = np.array(Y)
        colors = np.where(val == 1, 'g', 'r')
        plt.scatter(x1, x2, c = colors, marker='o', alpha=1)

        plt.axis([low_x1, hi_x1, low_x2, hi_x2])
        plt.xlabel('Parameter X1')
        plt.ylabel('Parameter X2')
       
        plt.show()

    @staticmethod
    def show_test_dot(parameters, mu, X_norm):

        X1 = []
        X2 = []
        colors = []

        for i in range(101):
            for j in range(0, 9001, 50):
                x1 = j
                x2 = i

                X = [[x1], [x2]]
                X = X - mu
                X = X / X_norm
                
                x1n = X[0, 0]
                x2n = X[1, 0]
                
                A2, cache = NNGradientDescent.forward_propagation(X, parameters)

                col = [(0,0,0)]
                if (A2 < 0.5):
                    col = (0.2,0,0)
                else:
                    col = (0,0.2,0)

                X1.append(x1)
                X2.append(x2)
                colors.append(col)

        plt.scatter(X1, X2, c = colors, marker='s', alpha=1)

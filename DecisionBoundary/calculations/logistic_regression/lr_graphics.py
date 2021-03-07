#from calculations.logistic_regression.gradient_descent import LRGradientDescent
from calculations.utils import Utils
import matplotlib.pyplot as plt
import numpy as np
import time

class LRGraphics:

    @staticmethod
    def plot_lines():
        x1, y1 = [0, 12, 24], [1, 4, 31]
        x2, y2 = [1, 10], [3, 2]
        plt.plot(x1, y1, marker = 'o')
        plt.show()

    @staticmethod
    def plot_dots(x1, x2, Y, w, b, mu, X_norm):

        low_x1 = 0
        hi_x1 = 9000

        low_x2 = 0
        hi_x2 = 100

        # Test desicion boundary
        LRGraphics.show_test_dot(w, b, mu, X_norm)

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
    def show_test_dot(w, b, mu, X_norm):
        #w1, w2, b = Graphics.get_example_params_set(3)
        w1, w2 = w[0, 0], w[1, 0]

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
                
                z = w1 * x1n + w2 * x2n + b
                a = Utils.sigmoid(z)
                col = [(0,0,0)]
                if (a < 0.5):
                    col = (0.2,0,0)
                else:
                    col = (0,0.2,0)

                X1.append(x1)
                X2.append(x2)
                colors.append(col)

        plt.scatter(X1, X2, c = colors, marker='s', alpha=1)

    @staticmethod
    def get_example_params_set(i):
        w1 = [-131.46865098, -56.36041488, -16.23860239, -56.26000083]
        w2 = [339.41113964, 155.63734803, 59.31011687, 155.40118534]
        b = [-0.3135431778843503, -0.17105946322050056, -0.06568856198013712, -0.17082181351389364]
        return w1[i], w2[i], b[i]


    def f(x1, w, b):
        w1 = w[0, 0]
        w2 = w[1, 0]
        return (-w1 * x1 - b) / w2

    @staticmethod
    def plot_dots2(x1, x2, Val):

        x = np.linspace(0, 10*np.pi, 100)
        y = np.sin(x)

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

        val = np.array(Val)
        colors = np.where(val == 1, 'g', 'r')
        ax.scatter(x1, x2, c = colors, marker='o', alpha=0.5)

        ax.axis([0, 9000, 0, 100])
        #ax.xlabel('Parameter X1')
        #ax.ylabel('Parameter X2')
        fig.canvas.draw()

        lx1, lx2 = [0, 9000], [31, 70]
        ax.plot(lx1, lx2, marker = 'o')
        fig.canvas.draw()
        
        time.sleep(3)

        # line1, = ax.plot(x, y, 'b-')

        # for phase in np.linspace(0, 10*np.pi, 100):
        #     line1.set_ydata(np.sin(0.5 * x + phase))
        #     fig.canvas.draw()

from random import *


class RandomDots:

    @staticmethod
    def generate_rand_coords():
        X1 = []
        X2 = []
        values = []

        for i in range(1000):
            x1 = randint(0, 9000)
            x2 = randint(0, 100)
            X1.append(x1)
            X2.append(x2)

            x2formula = 39/9000 * x1 + 31
            val = 1 if x2 > x2formula else 0
            values.append(val)
        return X1, X2, values
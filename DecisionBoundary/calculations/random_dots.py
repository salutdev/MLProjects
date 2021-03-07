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

    @staticmethod
    def generate_rand_coords_with_complex_distribution():
        X1 = []
        X2 = []
        values = []

        for i in range(1000):
            x1 = randint(0, 9000)
            x2 = randint(0, 100)
            X1.append(x1)
            X2.append(x2)

            x2formula1 = 39/9000 * x1 + 31
            x2formula2 = -1/40 * x1 + 175

            val = 1 if (x2 > x2formula1 and x2 > x2formula2) or (x2 < x2formula1 and x2 < x2formula2) else 0
            values.append(val)
        return X1, X2, values
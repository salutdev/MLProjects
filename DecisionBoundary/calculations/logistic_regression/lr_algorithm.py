from calculations.logistic_regression.lr_graphics import LRGraphics
from calculations.file_operations import FileOperations
from calculations.random_dots import RandomDots
from calculations.logistic_regression.lr_gradient_descent import LRGradientDescent


class LRAlgorithm:

    def calculate(self):

        X1, X2, Y = FileOperations.read_dots_coords("Points2.txt")
        # X1, X2, Y = RandomDots.generate_rand_coords()
        # FileOperations.write_dots_coords(X1, X2, Y)
        
        w, b, mu, X_norm = LRGradientDescent().calculate_params(X1, X2, Y)

        LRGraphics.plot_dots(X1, X2, Y, w, b, mu, X_norm)


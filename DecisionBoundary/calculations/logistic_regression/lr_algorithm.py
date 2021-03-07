from calculations.logistic_regression.lr_graphics import LRGraphics
from calculations.file_operations import FileOperations
from calculations.random_dots import RandomDots
from calculations.logistic_regression.lr_gradient_descent import LRGradientDescent


class LRAlgorithm:

    def calculate(self):

        # X1, X2, Y = FileOperations.read_dots_coords("points_linear_boundary.txt")
        # w, b, mu, X_norm = LRGradientDescent().calculate_params(X1, X2, Y)
        # LRGraphics.plot_dots(X1, X2, Y, w, b, mu, X_norm)

        self.calc_distribution2()

    def calc_distribution1(self):
        X1, X2, Y = FileOperations.read_dots_coords("points_linear_boundary.txt")
        w, b, mu, X_norm = LRGradientDescent().calculate_params(X1, X2, Y)
        LRGraphics.plot_dots(X1, X2, Y, w, b, mu, X_norm)

    def calc_distribution2(self):
        X1, X2, Y = FileOperations.read_dots_coords("points_linear_boundary_with_complex_distribution.txt")
        w, b, mu, X_norm = LRGradientDescent().calculate_params(X1, X2, Y)
        LRGraphics.plot_dots2(X1, X2, Y, w, b, mu, X_norm)

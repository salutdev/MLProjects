from calculations.file_operations import FileOperations
from calculations.neural_network.nn_gradient_descent import NNGradientDescent
from calculations.neural_network.nn_graphics import NNGraphics
from calculations.random_dots import RandomDots


class NNAlgorithm:
    
    def calculate(self):
        # #X1, X2, Y = RandomDots.generate_rand_coords()
        # X1, X2, Y = RandomDots.generate_rand_coords_with_complex_distribution()
        # FileOperations.write_dots_coords(X1, X2, Y, 'points_linear_boundary_with_complex_distribution.txt')

        self.calc_distribution2()

    def calc_distribution1(self):
        X1, X2, Y = FileOperations.read_dots_coords("points_linear_boundary.txt")
        parameters, mu, X_norm = NNGradientDescent().calculate_params(X1, X2, Y)
        NNGraphics.plot_dots(X1, X2, Y, parameters, mu, X_norm)

    def calc_distribution2(self):
        X1, X2, Y = FileOperations.read_dots_coords("points_linear_boundary_with_complex_distribution.txt")
        parameters, mu, X_norm = NNGradientDescent().calculate_params(X1, X2, Y)
        NNGraphics.plot_dots2(X1, X2, Y, parameters, mu, X_norm)
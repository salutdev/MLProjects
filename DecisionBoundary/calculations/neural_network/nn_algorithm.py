from calculations.file_operations import FileOperations
from calculations.neural_network.nn_gradient_descent import NNGradientDescent
from calculations.neural_network.nn_graphics import NNGraphics


class NNAlgorithm:
    
    def calculate(self):
        X1, X2, Y = FileOperations.read_dots_coords("Points2.txt")

        parameters, mu, X_norm = NNGradientDescent().calculate_params(X1, X2, Y)

        NNGraphics.plot_dots(X1, X2, Y, parameters, mu, X_norm)
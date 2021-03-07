class FileOperations:

    @staticmethod
    def read_dots_coords(fileName):

        X1 = []
        X2 = []
        Y = []

        file_path = FileOperations.get_file_path(fileName)

        with open(file_path, "r") as dots:
            lines = dots.read().splitlines()
            for line in lines:
                x1, x2, y = line.split()
                X1.append(int(x1))
                X2.append(int(x2))
                Y.append(int(y))

        return X1, X2, Y

    @staticmethod
    def write_dots_coords(X1, X2, values, file_name):

        file_path = FileOperations.get_file_path(file_name)

        with open(file_path, 'w') as fp:
            for i in range(len(X1)):
                fp.write(f'{X1[i]} {X2[i]} {values[i]}\n')

    @staticmethod
    def get_file_path(file_name):
        return f'training_sets/{file_name}'
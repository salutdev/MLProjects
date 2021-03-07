class FileOperations:

    @staticmethod
    def read_dots_coords(fileName):

        X1 = []
        X2 = []
        Y = []

        with open(fileName, "r") as dots:
            lines = dots.read().splitlines()
            for line in lines:
                x1, x2, y = line.split()
                X1.append(int(x1))
                X2.append(int(x2))
                Y.append(int(y))

        return X1, X2, Y

    @staticmethod
    def write_dots_coords(X1, X2, Val):

        with open('Points2.txt', 'w') as fp:
            for i in range(len(X1)):
                fp.write(f'{X1[i]} {X2[i]} {Val[i]}\n')
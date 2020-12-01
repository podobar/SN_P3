class UtilFunctions:
    @staticmethod
    def step(x: list):
        for i in range(len(x)):
            x[i] = 1.0 if x[i] >= 0 else -1.0
        return x
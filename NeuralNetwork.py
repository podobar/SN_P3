#https://github.com/yogonza524/HopfieldNetwork/tree/master/src
# https://www.yumpu.com/xx/document/read/43026957/141-metody-uczenia-sieci
import numpy as np
class HopfieldNetwork:

    def __init__(self, n: int, seed: int):
        self.n = n #size of vector
        self.seed = seed
        self.w = np.zeros((n,n)) #weights, initially 0.0 <-> 0.0
        self.patterns = list()

        return

    def train_hebb(self, training_set):
        self.patterns = training_set
        for x in training_set:
            self.w = np.add(self.w, np.subtract(np.outer(x,x), np.identity(len(x))))
        self.w = self.w * 1/self.n

    def train_oja(self, training_set, learning_rate: float):
        self.patterns = training_set
        for x in training_set:
            for i in range(self.n):
                for j in range(self.n):
                    V = np.sum(self.w[:,j])*x[j] #i-loop
                    self.w[i, j] += learning_rate * V * (x[i] - V * self.w[i, j])


    def test_sync(self):
        return
    def test_async(self):
        return
    # def update_sync_hebb(self, x):
    #     #Update in upper-right part first, then rewrite to lower-left
    #     for i in range(1, self.n):
    #         for j in range (i):
    #             self.w[i,j]
    #     return
    # def update_async(self, x):
    #     indices = np.arange(0, self.n, step=1)
    #     random.Random(self.seed).shuffle(indices)
    #     z = np.outer(x, x)
    #     z = np.subtract(z, np.identity(len(x)))
    #     for i in indices:

        return
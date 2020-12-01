#https://github.com/yogonza524/HopfieldNetwork/tree/master/src
# https://www.yumpu.com/xx/document/read/43026957/141-metody-uczenia-sieci
import numpy as np
from UtilFunctions import UtilFunctions
from ImageHelper import ImageHelper
from FileHelper import FileHelper
class HopfieldNetwork:

    def __init__(self, n: int, seed: int, shape):
        self.n = n #size of vector
        self.seed = seed
        self.w = np.zeros((n,n)) #weights, initially 0.0 <-> 0.0
        self.patterns = list()
        self.shape = shape

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
                    if i==j:
                        continue
                    V = np.sum(self.w[:,j])*x[j] #i-loop
                    self.w[i, j] += learning_rate * V * (x[i] - V * self.w[i, j])

    def test_sync(self, x):
        max_iterations = np.power(4, np.power(2, len(self.patterns)-1))
        i = 0
        is_loop = False
        found = False
        steps_made = list()
        steps_made.append(x)
        ImageHelper.save_image(data=FileHelper.reshape(data=x, shape=self.shape), index=i)
        while is_loop == False and found == False and i < max_iterations:
            x = UtilFunctions.step(np.multiply(self.w, x))
            if self.patterns.__contains__(x):
                found = True
            elif x in steps_made:
                is_loop = True
            steps_made.append(x)
            i += 1
            ImageHelper.save_image(data = FileHelper.reshape(data=x, shape=self.shape), index= i)
        if is_loop or i == max_iterations:
            return 1
        elif found:
            return 0

    def test_async(self, x:list): #TODO
        indices = np.arange(0, self.n, step=1)
        np.random.Random(self.seed).shuffle(indices)
        steps_made = list()
        while x not in steps_made:
            for i in indices:
                return #TODO
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


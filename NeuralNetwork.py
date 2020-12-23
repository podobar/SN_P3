#https://github.com/yogonza524/HopfieldNetwork/tree/master/src
# https://www.yumpu.com/xx/document/read/43026957/141-metody-uczenia-sieci
import numpy as np
import glob
import os

from Utils import Functions
from Utils import ImageHelper
from Utils import FileHelper

class HopfieldNetwork:
    def __init__(self, n: int, shape, seed: int = 100):
        self.n = n #size of vector
        self.shape = shape
        self.seed = seed

        self.w = np.zeros((n, n))  # weights, initially 0.0 <-> 0.0
        self.patterns = list()
        return

    def output(self):
        ret = np.zeros(self.n)
        for i in range(self.n):
            ret[i] = 1 if np.sum(self.w[i,:]) >= 0 else -1
        return ret

    def energy(self, x: list):
        ret_energy = 0.0
        for i in range(self.n):
            for j in range(self.n):
                ret_energy += -0.5 * self.w[i,j] * x[i] * x[j]
        return ret_energy

    def train_hebb(self, training_set):
        self.patterns = training_set
        for x in training_set:
            self.w = np.add(self.w, np.subtract(np.outer(x,x), np.identity(len(x))))

        self.w = self.w * 1/self.n
        ImageHelper.save_weights(self.w)
        return

    def train_oji(self, training_set, learning_rate: float = 0.001, epochs: int = 500):
        self.train_hebb(training_set)
        for epoch in range(epochs):
            w_old = self.w.copy()
            for x in training_set:
                for i in range(self.n):
                    V = np.sign(np.dot(self.w, x)[i])
                    for j in range(self.n):
                        self.w[i, j] += learning_rate * V * (x[i] - V * self.w[i, j])

            if np.linalg.norm(w_old - self.w) < 10e-15:
                break
        self.w -= np.diag(self.w)
        ImageHelper.save_weights(self.w)
        return

    def test_training_set(self, is_sync: bool=True):
        for pattern in self.patterns:
            self.test(pattern, is_sync, is_training_pattern=True)

    def test(self, x:list, is_sync: bool = True, is_training_pattern: bool = False):
        return self.__test_sync(x, is_training_pattern) if is_sync is True else self.__test_async(x, is_training_pattern)

    def __test_sync(self, x: list, is_training_pattern: bool = False):
        pattern_id = self.patterns.index(x) if is_training_pattern else None
        i=0
        energy_log = list()
        energy_log.append(self.energy(x))
        ImageHelper.save_image(data=FileHelper.reshape(data=x, shape=self.shape), index=i, pattern_id=pattern_id)
        while True:
            i+=1
            x = Functions.step(list(np.dot(self.w, x)))
            energy_log.append(self.energy(x))
            ImageHelper.save_image(data=FileHelper.reshape(data=x, shape=self.shape), index=i, pattern_id=pattern_id)
            if energy_log[i] >= energy_log[i-1] and i > 1:
    #Network reached stable state or is looped
                if x in self.patterns:
                    found_index = self.patterns.index(x)
                    test_result = f"Rozpoznano ({found_index}). "
                else:
                    test_result = "Nie rozpoznano. "
                break
        ImageHelper.save_energy_plot(data=energy_log, result=test_result, pattern_id=pattern_id)
        return x

    def __test_async(self, x: list, is_training_pattern: bool = False):
        pattern_id = self.patterns.index(x) if is_training_pattern else None
        i=0
        energy_log = list()
        energy_log.append(self.energy(x))
        ImageHelper.save_image(data=FileHelper.reshape(data=x, shape=self.shape), index=i, pattern_id=pattern_id)
        while True:
            np.random.seed(self.seed + i)
            i += 1
            for j in range(self.n):
                rand_index = np.random.randint(0, self.n)
                x[rand_index] = Functions.step(list(np.dot(self.w, x)))[rand_index]
            energy_log.append(self.energy(x))
            ImageHelper.save_image(data=FileHelper.reshape(data=x, shape=self.shape), index=i, pattern_id=pattern_id)
            if energy_log[i] == energy_log[i - 1]:
                # Network reached stable state or is looped
                if x in self.patterns:
                    found_index = self.patterns.index(x)
                    test_result = f"Rozpoznano ({found_index}). "
                else:
                    test_result = "Nie rozpoznano. "
                break
        ImageHelper.save_energy_plot(data=energy_log, result=test_result, pattern_id=pattern_id)
        return x
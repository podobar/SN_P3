import csv
import re
import numpy


class FileHelper:
    @staticmethod
    def load(file_path: str):
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            _data = list(reader)

        for index in range(len(_data)):
            _data[index] = [int(item) for item in _data[index]]

        return _data

    @staticmethod
    def get_size_from_name(name: str):
        match= re.search(pattern=r"([0-9]+)x([0-9]+)", string=name)
        if match:
            size = match.group().split('x')
            size = [int(item) for item in size]
        return size

    @staticmethod
    def reshape(data, shape: list):
        return numpy.reshape(data, newshape=shape)
import numpy
import matplotlib.pyplot as plt
import csv
import re
import time

class Functions:
    @staticmethod
    def step(x: list):
        for i in range(len(x)):
            x[i] = 1.0 if x[i] >= 0 else -1.0
        return x

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
            size.reverse()
        return size

    @staticmethod
    def reshape(data, shape: list):
        return numpy.reshape(data, newshape=shape)


class ImageHelper:
    @staticmethod
    def draw_images(images: list):
        if len(images) > 0 :
            fig = plt.figure()
            rows = 1
            columns = len(images)
            for i in range(len(images)):
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(images[i], cmap='gray', vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
            plt.show()


    @staticmethod
    def save_image(data, index, pattern_id = None):
        plt.imshow(data, cmap='gray', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        if pattern_id is None:
            plt.savefig(f'Results/{time.strftime("%m%d-%H%M")}_{index}.png')
        else:
            plt.savefig(f'Results/p{pattern_id}_{time.strftime("%m%d-%H%M")}_{index}.png')
        plt.close()

    @staticmethod
    def save_energy_plot(data: list, result: str, pattern_id = None):
        plt.plot(data)
        plt.title(result + ' Energia w zależności od iteracji (i)')
        plt.ylabel('Energia')
        plt.xlabel('i')
        if pattern_id is None:
            plt.savefig(f'Results/{time.strftime("%m%d-%H%M")}_energy.png')
        else:
            plt.savefig(f'Results/p{pattern_id}_{time.strftime("%m%d-%H%M")}_energy.png')
        plt.close()

    @staticmethod
    def save_weights(matrix):
        w_mat = plt.imshow(matrix, cmap='coolwarm')
        plt.colorbar(w_mat)
        plt.title("Wagi połączeń w sieci")
        plt.tight_layout()
        plt.savefig("Results/weights.png")
        plt.close()
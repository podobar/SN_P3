import matplotlib.pyplot as plt


class ImageHelper:
    @staticmethod
    def draw_image(data):
        plt.imshow(data, cmap='gray', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()
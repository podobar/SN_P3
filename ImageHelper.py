import matplotlib.pyplot as plt


class ImageHelper:
    @staticmethod
    def draw_image(data):
        plt.imshow(data, cmap='gray', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    @staticmethod
    def save_image(data, index):
        plt.imshow(data, cmap='gray', vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('iteration_{:04d}.png'.format(index))
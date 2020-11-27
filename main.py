from FileHelper import FileHelper as fh
from ImageHelper import ImageHelper as ih
from NeuralNetwork import HopfieldNetwork
import tkinter as tk
from tkinter import filedialog
# root = tk.Tk()
# root.title("Hopfield's network")
#
# #Global variables
# img_size = [0]
# training_data = [0]
# buttons = list()
# #
#
#
# def load_training_set_and_train():
#     train_file_path = filedialog.askopenfilename(initialdir="/Data", title="Choose training set", filetypes=(("csv files", "*.csv"),))
#     if train_file_path:
#         global img_size
#         global training_data
#
#         img_size = fh.get_size_from_name(train_file_path)
#         raw_data = fh.load(train_file_path)
#         #IF you want to see what you're training the network on, then toggle breakpoint on for below and use debug mode. Uncomment the code first of course.
#         # for arr in raw_data:
#         #     img = fh.reshape(data=arr, shape=img_size)
#         #     ih.draw_image(img)
#         training_data = raw_data
#         #TODO: Train here
#         #
#         clear_buttons()
#
#
# def clear_buttons():
#     global buttons
#     global img_size
#     for button in buttons:
#         button.destroy()
#     buttons.clear()
#     if len(img_size) == 2:
#         for i in range(img_size[0]):
#             for j in range(img_size[1]):
#                 b = tk.Button(root,
#                            text="",
#                            width=2,
#                            height=1,
#                            bg='black',
#                            command=lambda _index=i * img_size[1] + j: switch_background(index=_index)
#                            )
#                 b.grid(row=i, column=j)
#                 buttons.append(b)
#
#
# def switch_background(index: int):
#     buttons[index]['bg'] = 'white' if buttons[index]['bg'] == 'black' else 'black'
#     buttons[index].update()


if __name__ == '__main__':
    # __menu = tk.Menu(root)
    # root.config(menu=__menu)
    # __train_menu = tk.Menu(__menu)
    # __menu.add_cascade(label="Train", menu=__train_menu)
    # __train_menu.add_command(label='Choose training file and train', command=load_training_set_and_train)
    # root.mainloop()
    network = HopfieldNetwork(n=9, seed=1010, threshold=0.1)
    network.update_async([1,2,3])
    network.update_sync([1,2,3])
    print('done')
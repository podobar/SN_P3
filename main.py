from Utils import FileHelper as fh
from Utils import ImageHelper as ih
from NeuralNetwork import HopfieldNetwork
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.title("Hopfield's network")

#Global variables
img_size = None
training_data = None
buttons = list()
is_sync = True
train_method = 0 #Hebb
network = None
training_image_index = None
#


def load_training_set():
    train_file_path = filedialog.askopenfilename(initialdir="/Data", title="Choose training set", filetypes=(("csv files", "*.csv"),))
    if train_file_path:
        global img_size
        global training_data

        img_size = fh.get_size_from_name(train_file_path)
        raw_data = fh.load(train_file_path)
        training_data = raw_data
        clear_buttons()


def show_training_set():
    global training_data
    global img_size
    if training_data is not None and img_size is not None:
        images = list()
        for item in training_data:
            images.append(fh.reshape(data=item, shape=img_size))
        ih.draw_images(images)


def clear_buttons():
    global buttons
    global img_size
    for button in buttons:
        button.destroy()
    buttons.clear()
    if len(img_size) == 2:
        for i in range(img_size[0]):
            for j in range(img_size[1]):
                b = tk.Button(root,
                           text="",
                           width=2,
                           height=1,
                           bg='black',
                           command=lambda _index=i * img_size[1] + j: switch_background(index=_index)
                           )
                b.grid(row=i, column=j)
                buttons.append(b)


def switch_background(index: int):
    buttons[index]['bg'] = 'white' if buttons[index]['bg'] == 'black' else 'black'
    buttons[index].update()


def set_mode_async():
    global is_sync
    is_sync = False


def set_mode_sync():
    global is_sync
    is_sync = True


def set_train_Hebb():
    global train_method
    train_method = 0 #Hebb


def set_train_Oji():
    global train_method
    train_method = 1 #Oja


def get_list_from_buttons():
    global buttons
    ret = list()
    for button in buttons:
        ret.append(1 if button['bg'] == 'white' else -1)
    return ret


def set_buttons_from_list(_list):
    global buttons
    for i in range(len(_list)):
        buttons[i]['bg'] = 'white' if _list[i] == 1 else 'black'


def init_network():
    global train_method
    global img_size
    global training_data
    global is_sync
    global network
    global buttons
    if img_size is not None and len(img_size) == 2 and training_data is not None:
        network = HopfieldNetwork(n=img_size[0]*img_size[1], shape=img_size)
        if train_method == 0: #Hebb
            network.train_hebb(training_set=training_data)
        else:
            network.train_oji(training_set=training_data)
    clear_buttons()


def test_network():
    current_list = get_list_from_buttons()
    global network
    global is_sync
    if network is not None:
        set_buttons_from_list(network.test(x=current_list, is_sync=is_sync))


def test_network_on_training_set():
    global network
    global is_sync
    if network is not None:
        network.test_training_set(is_sync)


def load_next(event=None):
    global network
    global training_image_index
    if network is not None:
        if training_image_index is not None:
            training_image_index += 1
            training_image_index = training_image_index % len(network.patterns)
        else:
            training_image_index = 0
        set_buttons_from_list(network.patterns[training_image_index])


if __name__ == '__main__':
    __menu = tk.Menu(root)
    root.config(menu=__menu)
    __train_menu = tk.Menu(__menu)
    __train_mode_menu = tk.Menu(__menu)
    __test_menu = tk.Menu(__menu)
    __test_mode_menu = tk.Menu(__menu)

    __menu.add_cascade(label="Train", menu=__train_menu)
    __train_menu.add_command(label='Choose training data', command=load_training_set)
    __train_menu.add_command(label='Display training data', command=show_training_set)
    __train_menu.add_command(label='Create and train network', command=init_network)

    __menu.add_cascade(label='Training mode',menu=__train_mode_menu)
    __train_mode_menu.add_command(label='Train method - Hebb (default)', command=set_train_Hebb)
    __train_mode_menu.add_command(label='Train method - Oji', command=set_train_Oji)

    __menu.add_cascade(label="Test", menu=__test_menu)
    __test_menu.add_command(label='Test network on current map', command=test_network)
    __test_menu.add_command(label='Test network on training set', command=test_network_on_training_set)
    __test_menu.add_command(label='Load next image from training set', command=load_next)

    __menu.add_cascade(label='Testing mode', menu=__test_mode_menu)
    __test_mode_menu.add_command(label='Synchronous (default)', command=set_mode_sync)
    __test_mode_menu.add_command(label='Asynchronous', command=set_mode_async)
    root.bind("<space>", load_next)
    root.mainloop()

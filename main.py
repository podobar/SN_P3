from FileHelper import FileHelper as fh
from ImageHelper import ImageHelper as ih
import tkinter as tk
root = tk.Tk()
test_set_file_paths = [
    'Data/animals-14x9.csv',
    'Data/large-25x25.csv',
    'Data/large-25x25.plus.csv',
    'Data/large-25x50.csv',
    'Data/letters-14x20.csv',
    'Data/letters-abc-8x12.csv',
    'Data/OCRA-12x30-cut.csv',
    'Data/small-7x7.csv'
]


def __load_file():
    return


def __switch_background(index: int):
    __buttons[index]['bg'] = 'white' if __buttons[index]['bg'] == 'black' else 'black'
    __buttons[index].update()

if __name__ == '__main__':
    # train_file_path = test_set_file_paths[] #Insert index to show data set as black&white images
    # img_size = fh.get_size_from_name(train_file_path)
    # raw_data = fh.load(train_file_path)
    # for arr in raw_data:
    #     reshaped_data = fh.reshape(data=arr, shape=img_size)
    #     ih.draw_image(reshaped_data)
    # print('End')
    N = 5
    M = 10
    root.title("Hopfield's network")
    __buttons = list()
    __menu = tk.Menu(root)
    root.config(menu=__menu)
    __train_menu = tk.Menu(__menu)
    __menu.add_cascade(label="Train", menu=__train_menu)
    __train_menu.add_command(label='Choose training file and train', command=__load_file)

    for i in range(N):
        for j in range(M):
            b = tk.Button(root,
                       width=10,
                       height=10,
                       bg='black' if i % 2 == 0 else 'white',
                       command=lambda _index=i * M + j: __switch_background(index=_index)
                       )
            b.grid(row=i, column=j)
            __buttons.append(b)

    root.mainloop()
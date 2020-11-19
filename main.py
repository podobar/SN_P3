from FileHelper import FileHelper as fh
from ImageHelper import ImageHelper as ih

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

if __name__ == '__main__':
    train_file_path = test_set_file_paths[] #Insert index to show data set as black&white images
    img_size = fh.get_size_from_name(train_file_path)
    raw_data = fh.load(train_file_path)
    for arr in raw_data:
        reshaped_data = fh.reshape(data=arr, shape=img_size)
        ih.draw_image(reshaped_data)
    print('End')
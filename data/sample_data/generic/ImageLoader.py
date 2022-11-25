from utils.utils import load_data_loader, check_dims
import os
import numpy as np


def read_file(path):
    file_format = path.strip().split(".")[-1].strip()
    if path is None or path.split("/")[-1] == 'None':
        print(f"'None' cannot be a file name ({path})."
              f"The program consider this as NO FILE IS REQUIRED!")
        return None
    if os.path.exists(path):
        if file_format in ['npz', 'npy']:
            if file_format == 'npz':
                ds = np.load(path, allow_pickle=True)
                x, y = ds['data'], ds['targets']
            else:  # self.file_format == 'npy':
                x, y = np.load(path, allow_pickle=True)
            return check_dims(x), y
        print(f"{file_format} is not supported file format")
    else:
        print(f"{path} File Not Found!!!")
        print(f" Program will be terminated!!!")
        exit(0)


class DataLoader:
    def __init__(self, path=None):
        self.path = path
        self.loader = None

    @property
    def sample_data(self):
        if self.path is None:
            raise NotADirectoryError("Path to the data is None!")
        return read_file(self.path)

    def load(self, path, batch_size):
        data = read_file(path)
        if data is not None:
            x_train, y_train = data
            return load_data_loader(x_train, y_train, batch_size)

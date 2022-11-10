from utils.utils import load_data_loader, check_dims
import os
import numpy as np


# class DataReader:
#     def __init__(self, file_format):
#         self.file_format = file_format
#
#     def read_file(self, path):
#         if path is None or path.split("/")[-1] == 'None':
#             print(f"'None' cannot be a file name ({path})."
#                   f"The program consider this as NO FILE IS REQUIRED!")
#             return None
#         if os.path.exists(path):
#             if self.file_format in ['npz', 'npy']:
#                 if self.file_format == 'npz':
#                     ds = np.load(path, allow_pickle=True)
#                     x, y = ds['data'], ds['targets']
#                 else:  # self.file_format == 'npy':
#                     x, y = np.load(path, allow_pickle=True)
#                 return check_dims(x), y
#             print(f"{self.file_format} is not supported file format")
#         else:
#             print(f"{path} File Not Found!!!")
#             print(f" Program will be terminated!!!")
#             exit(0)


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


class ImageLoader:
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
            self.loader = load_data_loader(x_train, y_train, batch_size)


# class ImageLoader(DataReader):
#     def __init__(self, train_path, test_path, train_batch_size=None, test_batch_size=32, torch_mode=True):
#         path = train_path if train_path is not None else test_path
#         super().__init__(file_format=path.strip().split(".")[-1].strip())
#         self.train_path = train_path
#         self.test_path = test_path
#
#         self.train_loader = None
#         self.test_loader = None
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#         self.torch_mode = torch_mode
#
#     @property
#     def sample_data(self):
#         return self.read_file(self.train_path)
#
#     def lazy_init(self, train_batch_size, test_batch_size, torch_mode=True):
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#         self.torch_mode = torch_mode
#         if self.torch_mode:
#             self.load_data_loader()
#         else:
#             self.train_loader = IterLoader(self.train_path, self.train_batch_size)
#             self.train_loader_for_test = IterLoader(self.train_path, self.test_batch_size)
#
#     def load_data_loader(self):
#         data = self.read_file(self.train_path)
#         if data is not None:
#             x_train, y_train = data
#             self.train_loader = load_data_loader(x_train, y_train, self.train_batch_size)
#         data = self.read_file(self.test_path)
#         if data is not None:
#             x_test, y_test = data
#             self.test_loader = load_data_loader(x_test, y_test, self.test_batch_size)


class IterLoader(DataReader):
    def __init__(self, path, batch_size):
        super().__init__(path.strip().split(".")[-1].strip())
        self.path = path
        self.batch_size = batch_size
        self.batches = None
        self.max = 0
        self.n = 0
        x, y = self.read_file(self.path)
        n_samples = len(x)
        self.len = n_samples - (n_samples % self.batch_size)

    def __iter__(self):
        self.load()
        return self

    def __next__(self):
        try:
            x, y = self.batches[self.n]
        except:
            raise StopIteration()
        dl = load_data_loader(x, y, len(x))
        self.n += 1
        return next(iter(dl))

    def get_sample_data(self):
        x, y = self.batches[0]
        dl = load_data_loader(x, y, 1)
        return next(iter(dl))

    def load(self):
        samples, labels = self.read_file(self.path)
        n_samples = len(samples)
        n_batches = n_samples // self.batch_size

        samples_indices = np.arange(n_samples)
        np.random.shuffle(samples_indices)
        batches_idx = np.split(samples_indices[:self.batch_size * n_batches], n_batches)
        self.batches = []
        for batch_ind in batches_idx:
            x_batch = samples[batch_ind]
            y_batch = labels[batch_ind]
            self.batches.append([x_batch, y_batch])
        self.max = len(self.batches)
        self.n = 0
        self.len = self.max * self.batch_size

    def __len__(self):
        return self.len

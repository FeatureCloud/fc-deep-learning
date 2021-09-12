from .utils import load_dataloader
import os
import numpy as np


class DataLoader():
    def __init__(self, train_path, test_path, train_batch_size=None, test_batch_size=32, torch_mode=True):
        self.train_path = train_path
        self.test_path = test_path
        self.sample_data = read_file(self.train_path)
        self.train_loader = None
        self.train_loader_for_test = None
        self.test_loader = None
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.torch_mode = torch_mode

    def lazy_init(self, train_batch_size, test_batch_size, torch_mode=True):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.torch_mode = torch_mode
        if self.torch_mode:
            self.load_data_loader()
        else:
            self.train_loader = IterLoader(self.train_path, self.train_batch_size)
            self.train_loader_for_test = IterLoader(self.train_path, self.test_batch_size)

    def get_test_loader(self):
        x_test, y_test = read_file(self.test_path)
        if x_test is None or len(x_test) == 0:
            return None
        if self.torch_mode:
            return load_dataloader(x_test, y_test, self.test_batch_size)
        return IterLoader(self.test_path, self.test_batch_size)

    def load_data_loader(self):
        x_train, y_train = read_file(self.train_path)
        # self.sample_data = next(iter(load_dataloader(x_train[0], y_train[0], batch_size=1)))
        self.train_loader = load_dataloader(x_train, y_train, self.train_batch_size)
        self.train_loader_for_test = self.train_loader


def read_file(path):
    if os.path.exists(path):
        x, y = np.load(path, allow_pickle=True)
        x = np.array(list(x)).squeeze()
        return x, y
    elif path.split("/")[-1] == 'None':
        print(f"'None' cannot be a file name ({path})."
              f"The program consider this as NO FILE IS REQUIRED!")
        return None, None
    else:
        print(f"{path} File Not Found!!!")
        print(f" Program will be terminated!!!")
        exit(0)


class IterLoader:
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.batches = None
        self.max = 0
        self.n = 0
        x, y = read_file(self.path)
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
        dl = load_dataloader(x, y, len(x))
        self.n += 1
        return next(iter(dl))

    def get_sample_data(self):
        x, y = self.batches[0]
        dl = load_dataloader(x, y, 1)
        return next(iter(dl))

    def load(self):
        samples, labels = read_file(self.path)
        n_samples = len(samples)
        n_batches = n_samples // self.batch_size

        samples_indeces = np.arange(n_samples)
        np.random.shuffle(samples_indeces)
        batches_idx = np.split(samples_indeces[:self.batch_size * n_batches], n_batches)
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

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from scipy.stats import zscore


class CustomDataLoader:
    """
    RNA DataLoader supports sc-RNA dataset in a single CSV or TSV file while the class labels are in a specific column
    It should receive following kwargs:
        * sep: seperator character for the data file
        * label: column of the labels

    Attributes:
        path: str
        sep: str
        label: str
        file_format: str
        x: numpy.array
        y: numpy.array
        dataset: torch.utils.data.Dataset

    """

    def __init__(self, path=None, **kwargs):
        self.path = path
        self.sep = kwargs['sep']
        self.label = kwargs['label']
        self.file_format = None if self.path is None else self.path.strip().split(".")[-1].strip()
        self.x, self.y = None, None
        self.dataset = None

    @property
    def sample_data_loader(self):
        if self.path is None:
            raise NotADirectoryError("Path to the data is None!")
        self.read_file()
        self.dataset = RnaDataset(self.x, self.y)
        return DataLoader(self.dataset, 1, shuffle=True, num_workers=1)

    def load(self, path=None, batch_size=32):
        if path is not None:
            self.file_format = self.path.strip().split(".")[-1].strip()
            self.read_file()
        if self.x is not None and self.y is not None:
            self.dataset = RnaDataset(self.x, self.y)
            return DataLoader(self.dataset, batch_size, shuffle=True, num_workers=1)
        return None

    def read_file(self):
        if self.file_exits():
            data = pd.read_csv(self.path, sep=self.sep)
            self.x = zscore(data.drop(columns=self.label).values, axis=1)
            self.y = data[self.label].values

    def file_exits(self):
        if self.path is None or self.path.split("/")[-1] == 'None':
            print(f"'None' cannot be a file name ({self.path})."
                  f"The program consider this as NO FILE IS REQUIRED!")
            return False
        if os.path.exists(self.path):
            if self.file_format not in ['csv', 'tsv']:
                print(f"{self.file_format} file format is not supported!")
                return False
            return True
        print(f"{self.path} File Not Found!!!")
        print(f" Program will be terminated!!!")
        return False


class RnaDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].astype("float32")
        target = self.label[idx].astype("long")
        return data, target

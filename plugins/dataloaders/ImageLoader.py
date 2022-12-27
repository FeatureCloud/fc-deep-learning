import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image


class CustomDataLoader:
    def __init__(self, path=None):
        self.path = path
        self.file_format = None if self.path is None else self.path.strip().split(".")[-1].strip()
        self.x, self.y = None, None
        self.dataset = None

    @property
    def sample_data_loader(self):
        if self.path is None:
            raise NotADirectoryError("Path to the data is None!")
        self.read_file()
        self.x = np.array(list(self.x)).squeeze()
        if self.x.shape[-1] > 3:  # channel first
            self.x = np.moveaxis(self.x, -1, 1)
        self.dataset = FromNumpyDataset(self.x, self.y, transforms.Compose([transforms.ToTensor()]))
        return DataLoader(self.dataset, 1, shuffle=True, num_workers=1)

    def load(self, path=None, batch_size=32):
        if path is not None:
            self.file_format = self.path.strip().split(".")[-1].strip()
            self.read_file()
        if self.x is not None and self.y is not None:
            transform = self.get_normalized_transform()
            self.dataset = FromNumpyDataset(self.x, self.y, transform)
            return DataLoader(self.dataset, batch_size, shuffle=True, num_workers=1)
        return None

    def read_file(self):
        if self.file_exits():
            if self.file_format == 'npz':
                ds = np.load(self.path, allow_pickle=True)
                self.x, self.y = ds['data'], ds['targets']
            else:  # self.file_format == 'npy':
                self.x, self.y = np.load(self.path, allow_pickle=True)
            self.x = np.array([pic.astype("float32") for pic in self.x], dtype="float32")

    def file_exits(self):
        if self.path is None or self.path.split("/")[-1] == 'None':
            print(f"'None' cannot be a file name ({self.path})."
                  f"The program consider this as NO FILE IS REQUIRED!")
            return False
        if os.path.exists(self.path):
            if self.file_format not in ['npz', 'npy']:
                print(f"{self.file_format} is not supported file format")
                return False
            return True
        print(f"{self.path} File Not Found!!!")
        print(f" Program will be terminated!!!")
        return False

    def get_normalized_transform(self):
        if self.x.ndim > 3:
            mean = [np.mean(self.x[:, :, :, ch]) for ch in range(self.x.shape[-1])]
            std = [np.std(self.x[:, :, :, ch]) for ch in range(self.x.shape[-1])]
        else:
            mean, std = [self.x.mean()], [self.x.std()]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        return transform


class FromNumpyDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.features = x_train
        self.labels = y_train
        self.transform = transform

    def __getitem__(self, index):
        np_arr = np.array(self.features[index]).astype("float32")
        y = self.labels[index]
        img = Image.fromarray(np_arr)
        if self.transform is not None:
            img = self.transform(np.array(img))
        return img, y

    def __len__(self):
        return self.features.shape[0]

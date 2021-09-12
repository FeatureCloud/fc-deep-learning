"""
    FeatureCloud DeepLearning Application

    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn


def get_base_model_config(stats, hyper_parameters):
    """

    Parameters
    ----------
    stats: dict
    hyper_parameters: dict

    Returns
    -------
    config: dict
    """
    # if hyper_parameters['model'] == "CNN":
    if True:
        config = {"n_channels": stats['n_channels'],
                  "img_width": stats['width'],
                  "n_classes": hyper_parameters["n_classes"],
                  }
    return config


def load_data_loader(x, y, batch_size):
    """

    Parameters
    ----------
    x: numpy.array
        features
    y: list
        labels
    batch_size: int

    Returns
    -------
    data_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = FromNumpyDataset(x, y, transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=1)
    return data_loader


def design_model(config, sample):
    ds = load_data_loader(sample[0], sample[1], batch_size=1)
    sample_data = next(iter(ds))[0]
    del ds
    layer_default = {}
    layers = []
    layer_counter = {}
    names = []
    params = []
    for item in config:
        layer_type = item['type']
        layer_counter[layer_type] = layer_counter[layer_type] + 1 if layer_type in layer_counter else 1
        layers.append(getattr(nn, layer_type))
        names.append(f'{layer_type.lower()}{layer_counter[layer_type]}')
        param = item['param'] if 'param' in item else {}
        param = get_param_value_from_data(param, sample_data, layers[:-1], names[:-1], params)
        layer_default[layer_type] = {**layer_default[layer_type], **param} if layer_type in layer_default else param
        params.append({**layer_default[layer_type], **param})

    model_gen = get_model_generator()

    return model_gen, {'layers': layers, 'layer_names': names, 'params': params}


def get_model_generator():
    class Model(nn.Module):
        def __init__(self, layers, layer_names, params):
            self.layer_names = layer_names
            super(Model, self).__init__()
            for layer, name, param in zip(layers, layer_names, params):
                setattr(self, name, layer(**param))

        def forward(self, x):
            for name in self.layer_names:
                x = getattr(self, name)(x)
            return x

    return Model


def get_param_value_from_data(raw_params, sample, *args):
    if len(raw_params) > 0:
        if 'in_features' in raw_params:
            if raw_params['in_features'] == 'None':
                model_gen = get_model_generator()
                model = model_gen(*args)
                n_flat_input_features = model(sample).size(1)
                raw_params['in_features'] = n_flat_input_features
        if 'in_channels' in raw_params:
            if raw_params['in_channels'] == 'None':
                raw_params['in_channels'] = sample.size(1)
    return raw_params


def num_flat_features(x):
    """ calculating the size of x for all dimensions
        except the first one (batch).

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    num_features : int
        number of elements of x(except first dim.)
    """
    size = x.size()[1:]  #
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class FromNumpyDataset(Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.features = x_train
        self.labels = y_train
        self.transform = transform

    def __getitem__(self, index):
        np_arr = np.array(self.features[index]).astype(f"float32")
        y = self.labels[index]
        img = Image.fromarray(np_arr)
        if self.transform is not None:
            img = self.transform(np.array(img))
        return img, y

    def __len__(self):
        return self.features.shape[0]

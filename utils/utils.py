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
import numpy as np
import torch.nn as nn
from models.pytorch import models
import importlib.util
import sys
from utils.pytorch import DataLoader as SupportedLoaders
from itertools import compress
import torch.optim as optim


def get_dataloader(name):
    """ Loading the class of some module that maybe implemented in models or uploaded by the user
    The Costume DataLoader should be always named `CustomDataLoader`
    And should have followings:
        * `sample_data` attribute to get a couple of samples
        * `load` method which gets the file_path and batch-size and returns the data loader


    Parameters
    ----------
    name: str
        name of the module (or.py file including it)

    Returns
    -------

    """
    return get_custom_module(module=name, existing=SupportedLoaders, class_name='CustomDataLoader')


def get_metrics(name, package):
    """ return the class of metrics that can be provided in one of the following ways:
        1. supported metrics in torchmetrics
        2. custom metrics that should be implemented in a similar fashion to torchmetrics
            more info: https://torchmetrics.readthedocs.io/en/stable/pages/implement.html

    Parameters
    ----------
    name: str
        name of the metric
         for torchmetrics, it should be the same importable name from torchmetrics
    package: str
        any torchmetrics' package that provides the metric. e.g., torchmetrics.classification

    Returns
    -------
    torchmetrics.Metric

    """
    return get_custom_module(module=name,
                             existing=importlib.import_module(package),
                             class_name='CustomMetric')


def get_loss_func(name):
    """ return the class of loss that can be provided in one of the following ways:
        1. supported losses in torch.nn
        2. custom loss that should be implemented as a class (methods are not supported)
            more info: https://neptune.ai/blog/pytorch-loss-functions

    Parameters
    ----------
    name: str
        name of the loss function
        for implemented loss functions in nn, it should be the same importable name from nn

    Returns
    -------
    nn.Module: loss class

    """
    return get_custom_module(module=name,
                             existing=nn,
                             class_name='CustomLoss')


def get_optimizer(name):
    """ return the class of optimizer that can be provided in one of the following ways:
        1. supported optimizers in torch.optim
        2. custom optimizer that should be implemented as a class

    Parameters
    ----------
    name: str
        name of the optimizer
        for implemented optimizer in torch.optim, it should be the same importable name from torch.optim

    Returns
    -------
    torch.optim.optimizer.Optimizer

    """
    return get_custom_module(module=name,
                             existing=optim,
                             class_name='CustomOptimizer')


def get_custom_module(module, existing, class_name=None):
    """ Loading the class of some module that maybe implemented in models or uploaded by the user
        This method supports following custom modules:
            * DataLoader
            * model
            * loss function
            * optimizer
    Parameters
    ----------
    module: str
        name of the module (or.py file including it)
    existing: python class
    class_name: str
        the name of the class that module contains and should be imported

    Returns
    -------

    """
    if hasattr(existing, module):
        return getattr(existing, module)
    if '.py' in module:
        dl = f"{module.split('.')[0]}.{class_name}"
        spec = importlib.util.spec_from_file_location(dl, f"/mnt/input/{module}")
        foo = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = foo
        spec.loader.exec_module(foo)
        return getattr(foo, class_name)
    return None


def design_model(config, data_loader):
    if 'name' in config:
        name = config.pop('name')
        return lambda: get_custom_module(module=name, existing=models, class_name='Model')(**config), {}
    sample_data = next(iter(data_loader))[0]
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


def to_list(np_array):
    if not isinstance(np_array, (np.ndarray, list)):
        if isinstance(np_array, np.float32):
            return np_array.item()
        return np_array
    lst_arr = []
    for item in np_array:
        if isinstance(np_array, (np.ndarray, list)):
            non_np = to_list(item)
        else:
            non_np = item
        lst_arr.append(non_np)
    return lst_arr


def to_numpy(lst):
    if isinstance(lst, list):
        np_arr = []
        for item in lst:
            np_item = item
            if isinstance(item, list):
                np_item = to_numpy(item)
            np_arr.append(np_item)
        try:
            return np.array(np_arr, dtype='float32')
        except:
            return np.array(np_arr, dtype='object')
    return lst


def average_weights(params):
    n_clients = len(params)
    n_splits = len(params[0])
    global_weights = [np.array(params[0][0][0], dtype='object') * 0] * n_splits
    total_n_samples = [0] * n_splits
    for client_models in params:
        for model_counter, (weights, n_samples) in enumerate(client_models):
            global_weights[model_counter] += np.array(weights, dtype='object') * n_samples
            total_n_samples[model_counter] += n_samples
    updated_weights = []
    for counter, (w, n) in enumerate(zip(global_weights, total_n_samples)):
        updated_weights.append(w / n)
    return updated_weights


def inject_root_path_to_clients_dir(client_dir, dirs, input=True):
    root_dir = f"/mnt/{'input' if input else 'output'}"
    injected_dirs = [d.replace(root_dir, f"{root_dir}/{client_dir}") for d in dirs]
    return injected_dirs


def get_path_to_central_test_output_files():
    central_pred_file = "/mnt/output/central_pred.csv"
    central_target_file = "/mnt/output/central_target.csv"
    return central_pred_file, central_target_file


def remove_converged_models(weights, state_dict, train_loaders, test_loaders, converged):
    if any(converged):
        weights = list(compress(weights, converged))
        state_dict = list(compress(state_dict, converged))
        train_loaders = list(compress(train_loaders, converged))
        test_loaders = list(compress(test_loaders, converged))
    return weights, state_dict, train_loaders, test_loaders

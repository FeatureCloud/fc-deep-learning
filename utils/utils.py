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
import torch
from models.pytorch import models
import importlib.util
import importlib
import sys
import os
from itertools import compress
import torch.optim as optim


def device_generator():
    for i in range(torch.cuda.device_count()):
        yield torch.device(f"cuda:{i}")


devices = device_generator()
def set_device(device):
    if torch.cuda.is_available() and device.strip().lower() == 'gpu':
        return next(device)
    return torch.device("cpu")



def is_native():
    path_prefix = os.getenv("PATH_PREFIX")
    if path_prefix:
        return False
    return True


def get_root_path(input=True):
    if input:
        return f"mnt/input"
    return f"mnt/output"


def get_trainer(name, root_dir):
    return get_custom_module(module=name,
                             existing=importlib.import_module('utils.pytorch.DeepModel'),
                             root_dir=root_dir,
                             class_name='CustomTrainer')


def get_aggregator(name, root_dir):
    """

    Parameters
    ----------
    name: str
        The name of the aggregator

    Returns
    -------


    """
    return get_custom_module(module=name,
                             existing=importlib.import_module('utils.pytorch.optimizer'),
                             root_dir=root_dir,
                             class_name='CustomAggregator')


def get_dataloader(name, root_dir):
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
    return get_custom_module(module=name,
                             existing=importlib.import_module('utils.pytorch.DataLoader'),
                             root_dir=root_dir,
                             class_name='CustomDataLoader')


def get_metrics(name, package, root_dir):
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
                             root_dir=root_dir,
                             class_name='CustomMetric')


def get_loss_func(name, root_dir):
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
                             root_dir=root_dir,
                             class_name='CustomLoss')


def get_optimizer(name, root_dir):
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
                             root_dir=root_dir,
                             class_name='CustomOptimizer')


def get_custom_module(module, existing, root_dir, class_name=None):
    """ Loading the class of some module that maybe implemented in models or uploaded by the user
        This method supports following custom modules:
            * DataLoader
            * model
            * loss function
            * optimizer
    Parameters
    ----------
    root_dir
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
        spec = importlib.util.spec_from_file_location(dl, f"{root_dir}/{module}")
        foo = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = foo
        spec.loader.exec_module(foo)
        return getattr(foo, class_name)
    return None


def design_architecture(config, data_loader, root_dir):
    if 'name' in config:
        name = config.pop('name')
        return lambda: get_custom_module(module=name, existing=models, class_name='Model', root_dir=root_dir)(**config), {}
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


def to_numpy(lst, dtype='float32', skip_obj_dtype=False):
    if isinstance(lst, list):
        np_arr = []
        for item in lst:
            np_item = item
            if isinstance(item, list):
                np_item = to_numpy(item, dtype, skip_obj_dtype)
            np_arr.append(np_item)
        try:
            return np.array(np_arr, dtype=dtype)
        except:
            print("here")
            if skip_obj_dtype:
                return np_arr
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
    # root_dir = f"{get_root_path(input)}/mnt/{'input' if input else 'output'}"
    root_dir = get_root_path(input)
    injected_dirs = [d.replace(root_dir, f"{root_dir}/{client_dir}") for d in dirs]
    return injected_dirs


def get_path_to_central_test_output_files():
    central_pred_file = f"{get_root_path(input=False)}/central_pred.csv"
    central_target_file = "get_root_path(input=False)/central_target.csv"
    return central_pred_file, central_target_file


def remove_converged_models(update_dict, backup, train_loaders, test_loaders):
    converged = update_dict['stoppage']
    if any(converged):
        update_dict = {k: list(compress(w, converged)) for k, w in update_dict.items()}
        backup = list(compress(backup, converged))
        train_loaders = list(compress(train_loaders, converged))
        test_loaders = list(compress(test_loaders, converged))
    return update_dict, backup, train_loaders, test_loaders


def unpack(global_updates, update_schema):
    weight, gradient, config, stoppage, cv = update_schema
    updates = ['weights', 'gradient', 'config', 'stoppage']
    g_weight, g_gradient, g_config, g_stoppage = None, None, None, None

    if weight:
        if config & stoppage:
            g_weight, g_config, g_stoppage = global_updates
        if config:
            g_weight, g_config = global_updates
        elif stoppage:
            g_weight, g_stoppage = global_updates
    elif gradient:
        if config & stoppage:
            g_gradient, g_config, g_stoppage = global_updates
        elif config:
            g_gradient, g_config = global_updates
        else:
            g_gradient, g_stoppage = global_updates
    return dict(zip(updates, [g_weight, g_gradient, g_config, g_stoppage]))


def interpret_global_updates(global_updates, update_schema):
    weight, gradient, config, stoppage, cv = update_schema
    expected_elements = [weight, gradient, config]
    expected_numl = sum(expected_elements)
    n_upd_elments = len(global_updates)
    if expected_numl > 1:
        assert n_upd_elments == expected_numl, f"Not the same number of elements in the global updates:" \
                                               f" Expected: {expected_numl}({expected_elements}) <> " \
                                               f"number of update elements: {n_upd_elments}:"

    if cv:
        cv_len = [len(u_elm) for u_elm in global_updates]
        cv_folds = sum(cv_len) // len(cv_len)
        assert all([l == cv_folds for l in cv_len]), f"Not all elements have the same cv fold: {cv_folds}"


def cv_first(updates):

    if updates['weights'] is not None:
        n_folds = len(updates['weights'])
    else:
        n_folds = len(updates['gradients'])

    transposed = [{} for _ in range(n_folds)]
    if updates.get('weights', None) is None:
        for i in range(n_folds):
            transposed[i]['weights'] = None
    else:
        for i, p in enumerate(updates['weights']):
            transposed[i]['weights'] = p
    if updates.get('gradients', None) is None:
        for i in range(n_folds):
            transposed[i]['gradients'] = None
    else:
        for i, p in enumerate(updates['gradients']):
            transposed[i]['gradients'] = p

    if updates.get('config', None) is None:
        for i in range(n_folds):
            transposed[i]['config'] = None
    else:
        if len(updates['config']) == n_folds:
            for i, c in enumerate(updates['config']):
                transposed[i]['config'] = c
        else:
            for i in range(n_folds):
                transposed[i]['config'] = updates['config']

    if updates.get('stoppage', None) is None:
        for i in range(n_folds):
            transposed[i]['stoppage'] = None
    else:
        for i, s in enumerate(updates['stoppage']):
            transposed[i]['stoppage'] = s

    return transposed
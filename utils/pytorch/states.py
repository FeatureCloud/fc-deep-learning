from abc import ABC

import numpy as np
from FeatureCloud.app.engine.app import AppState, LogLevel
from FeatureCloud.app.engine.app import State as op_state
from sqlalchemy.sql.functions import user

from CustomStates import ConfigState
from utils.pytorch.DataLoader import ImageLoader
from utils.pytorch import DataLoader
from utils.utils import design_model, load_module, to_list, to_numpy
from utils.pytorch.DeepModel import Model
from utils.pytorch.ClientModels import ClientModels
from itertools import compress
import pandas as pd
from copy import deepcopy
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Initialization(ConfigState.State, ABC):
    """
    Read input data
    read config files

    """

    def run(self) -> str or None:

        self.update(state=op_state.RUNNING)
        self.update(message="Reading the config file....")
        self.read_config()
        self.lazy_init()
        self.finalize_config()
        self.store('smpc_used', self.config.get('use_smpc', False))
        self.store('iteration', 0)
        device = torch.device('cuda' if torch.cuda.is_available() and self.config['gpu'] else 'cpu')
        self.read_input(device)
        if self.is_coordinator:
            data_to_send = [self.load('client_model').get_weights()]
            self.broadcast_data(data=data_to_send, send_to_self=False)

    def read_input(self, device):
        train_loaders, test_loaders = [], []

        model, client_model = None, None
        if self.load('input_files')['test'] is None and self.load('input_files')['central_test'] is None:
            self.log("There is no test data provided", LogLevel.ERROR)
            self.update(message="no test data", state=op_state.ERROR)
        self.log(f"Getting sample shape from {self.load('input_files')['train'][0]} dataset")

        dl_class = load_module(impl_models=DataLoader,
                               module=self.config['train_config']['data_loader'],
                               sub_module='DataLoader')
        if dl_class is None:
            self.log(f"module {self.config['train_config']['data_loader']} neither is found in implemented models nor "
                     f"in `/mnt/input` directory", LogLevel.ERROR)
            self.update(message="module not found", state=op_state.ERROR)
        dl = dl_class(path=self.load('input_files')['train'][0])
        sample_data = dl.sample_data
        for train_path, test_path in zip(self.load('input_files')['train'], self.load('input_files')['test']):
            model_class, config = design_model(deepcopy(self.config['model']), sample_data)
            train_all = self.config["fed_hyper_params"]["federated_model"].strip().lower() == "fedavg"
            if model is None:
                model = Model(model_class, config, self.config['train_config'], device)
            train_loaders.append(dl.load(train_path, model.batch_size))
            test_loaders.append(dl.load(test_path, model.test_batch_size))
            if client_model is None:
                client_model = ClientModels(model, train_all, self.config["fed_hyper_params"]["batch_count"])
                # if self.coordinator:
                #     client_model = ClientModels(model, train_all,
                #                                 self.config["fed_hyper_params"]["batch_count"])
                # else:
                #     client_model = ClientModels(model, train_all,
                #                                 self.config["fed_hyper_params"]["batch_count"])

        if self.is_coordinator:
            self.store('test_loader', dl.load(self.load('input_files')['central_test'][0], model.test_batch_size))
        self.store("fed_hyper_params", self.config["fed_hyper_params"])
        self.store('n_splits', len(train_loaders))
        self.store("state_dict", [client_model.get_optimizer_params()] * self.load('n_splits'))
        self.store('client_model', client_model)
        self.store('train_loaders', train_loaders)
        self.store('test_loaders', test_loaders)


class LocalUpdate(AppState, ABC):
    """ Local Model training
        Input:
            Model weights(Coordinator already has it)
            App statuses: {Converged: True/False }
    """

    def run(self) -> str or None:
        client_model = self.load('client_model')
        train_loaders = self.load('train_loaders')
        test_loaders = self.load('test_loaders')
        state_dict = self.load('state_dict')
        self.update(message=f"#{self.load('iteration') + 1}: Waiting for Coordinator")
        weights, converged = self.get_global_parameters(client_model, n_splits=self.load('n_splits'))

        if all(converged):
            return 'Converged'

        weights, state_dict, train_loaders, test_loaders = \
            self.remove_converged_models(weights, state_dict, train_loaders, test_loaders, converged)
        data_to_send = self.local_computation(client_model, train_loaders, test_loaders, weights, state_dict)
        self.send_data_for_aggregation(data_to_send)

    def send_data_for_aggregation(self, data):
        data_to_send = data
        if self.load('smpc_used'):
            data_to_send = to_list([[np.array(d, dtype='object') * w, w] for d, w in data])
        self.send_data_to_coordinator(data_to_send, use_smpc=self.load('smpc_used'))

    def get_global_parameters(self, client_model, n_splits):
        """

        Parameters
        ----------
        client_model
        n_splits

        Returns
        -------
        weights
        state_dict
        converged

        """
        converged = [False] * n_splits
        if self.is_coordinator:
            if self.load('iteration') == 0:
                weights = [client_model.get_weights()] * n_splits
                return weights, converged
            return self.load('weights'), self.load('converged')
        if self.load('iteration') == 0:
            weights = self.await_data(unwrap=True) * n_splits
            return weights, converged
        weights, converged = self.await_data(unwrap=True)
        return weights, converged

    def remove_converged_models(self, weights, state_dict, train_loaders, test_loaders, converged):
        if any(converged):
            weights = list(compress(weights, converged))
            state_dict = list(compress(state_dict, converged))
            train_loaders = list(compress(train_loaders, converged))
            test_loaders = list(compress(test_loaders, converged))
        return weights, state_dict, train_loaders, test_loaders

    def local_computation(self, client_model, train_loaders, test_loaders, weights, state_dicts):
        self.store('iteration', self.load('iteration') + 1)
        self.update(message=f"#{self.load('iteration')}: Local Training")
        new_parameters, new_state_dicts, trained_samples = [], [], []
        for counter, (tr_dl, test_dl, w, sd) in enumerate(zip(train_loaders, test_loaders, weights, state_dicts)):
            self.log(f"Iteration {self.load('iteration')}: Update model #{counter}")

            client_model.update(tr_dl, w, sd, verbose=True)
            if test_dl is not None:
                client_model.evaluate(test_dl)
            new_parameters.append(client_model.get_weights())
            trained_samples.append(client_model.num_trained_samples)
            new_state_dicts.append(client_model.get_optimizer_params())
        self.store('state_dict', new_state_dicts)
        return list(zip(new_parameters, trained_samples))


class GlobalAggregation(AppState, ABC):
    def run(self) -> str or None:
        self.update(message=f"#{self.load('iteration')}: Waiting for others")
        received_params = self.gather_local_models()
        self.update(message=f"#{self.load('iteration')}: Aggregation")
        global_weights, stopping_criteria = self.global_aggregation(received_params)
        data_to_send = [global_weights, stopping_criteria]
        self.broadcast_data(data_to_send, send_to_self=False)
        if all(stopping_criteria):
            return "Converged"

    def global_aggregation(self, params):
        client_model = self.load('client_model')
        if self.load('smpc_used'):
            global_weights = [to_numpy(model) / total_n_sample for model, total_n_sample in params]
        else:
            global_weights = self.average_weights(params, client_model)
        stopping_criteria = self.test_aggregated_models(global_weights, client_model)

        self.store('weights', global_weights)
        self.store('converged', stopping_criteria)
        return global_weights, stopping_criteria

    def average_weights(self, params, client_model):
        global_weights = [np.array(client_model.get_weights(), dtype='object') * 0] * self.load('n_splits')
        total_n_samples = [0] * self.load('n_splits')
        for client_models in params:
            for model_counter, (weights, n_samples) in enumerate(client_models):
                global_weights[model_counter] += np.array(weights, dtype='object') * n_samples
                total_n_samples[model_counter] += n_samples
        updated_weights = []
        for counter, (w, n) in enumerate(zip(global_weights, total_n_samples)):
            updated_weights.append(w / n)
        return updated_weights

    def test_aggregated_models(self, global_weights, client_model):
        iteration = self.load('iteration')
        max_iter = self.load('fed_hyper_params')["max_iter"]
        test_set = self.load('test_loader')
        stopping_criteria = []
        for counter, w in enumerate(global_weights):
            self.update(message=f"Global aggregation ")
            if test_set is not None:
                self.update(message=f"#{self.load('iteration')}: Test G model")
                self.log(f"Iteration #{self.load('iteration')}: Testing Global model #{counter}")
                client_model.set_weights(w)
                loss, acc = client_model.evaluate(test_set)
                self.log(f"Iteration #{self.load('iteration')}: Global aggregation of model #{counter}:"
                         f" Acc={acc:.2f} Loss={loss:.2f}")
            # TODO: stopping criterion!!!
            stopping_criteria.append(iteration >= max_iter)
        return stopping_criteria

    def gather_local_models(self):
        if self.load('smpc_used'):
            return self.await_data(is_json=True)
        return self.gather_data()


class WriteResults(AppState, ABC):
    def run(self) -> str or None:
        self.update(message=f"Writing Results")
        client_model = self.load('client_model')
        if self.is_coordinator:
            if self.load('test_loader') is not None:
                self.log(f"Writing the results for centralized test")
                y_pred, y_true = client_model.predict(self.load('test_loader'))
                pd.DataFrame(y_pred, columns=['y_pred']).to_csv(self.load('output_files')['central_pred'][0],
                                                                index=None)
                pd.DataFrame(y_true, columns=['y_true']).to_csv(self.load('output_files')['central_target'][0],
                                                                index=None)
        for counter, dl in enumerate(self.load('test_loaders')):
            if dl is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                y_pred, y_true = client_model.predict(dl)
                pd.DataFrame(y_pred, columns=['y_pred']).to_csv(self.load('output_files')['pred'][counter], index=None)
                pd.DataFrame(y_true, columns=['y_true']).to_csv(self.load('output_files')['target'][counter],
                                                                index=None)
        self.update(message="Finished!")

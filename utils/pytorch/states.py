from abc import ABC

import numpy as np
from FeatureCloud.app.engine.app import AppState, LogLevel
from FeatureCloud.app.engine.app import State as op_state
from CustomStates import ConfigState
from utils.pytorch.DataLoader import DataLoader
from utils.utils import design_model
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
        self.store('iteration', 0)
        device = torch.device('cuda' if torch.cuda.is_available() and self.config['gpu'] else 'cpu')
        self.read_input(device)
        if self.is_coordinator:
            # data_to_send = [self.load('client_model').get_weights(), self.load('client_model').get_optimizer_params()]
            data_to_send = [self.load('client_model').get_weights()]
            self.broadcast_data(data=data_to_send, send_to_self=False)

    def read_input(self, device):
        data_loaders = []

        model = None
        client_model = None
        if self.load('input_files')['test'] is None and self.load('input_files')['central_test'] is None:
            self.log("There is no test data provided", LogLevel.ERROR)

        for train, test in zip(self.load('input_files')['train'], self.load('input_files')['test']):
            data_loaders.append(DataLoader(train, test))
            model_class, config = design_model(deepcopy(self.config['model']),
                                               data_loaders[-1].sample_data)
            train_all = self.config["fed_hyper_params"]["federated_model"].strip().lower() == "fedavg"
            if model is None:
                model = Model(model_class, config, self.config['train_config'], device)
            data_loaders[-1].lazy_init(model.batch_size,
                                       model.test_batch_size,
                                       self.config['train_config']['torch_loader']
                                       )
            if client_model is None:
                client_model = ClientModels(model, train_all, self.config["fed_hyper_params"]["batch_count"])
                # if self.coordinator:
                #     client_model = ClientModels(model, train_all,
                #                                 self.config["fed_hyper_params"]["batch_count"])
                # else:
                #     client_model = ClientModels(model, train_all,
                #                                 self.config["fed_hyper_params"]["batch_count"])

        if self.is_coordinator:
            self.load_central_testset(model)
        self.store("fed_hyper_params", self.config["fed_hyper_params"])
        self.store("state_dict", [client_model.get_optimizer_params()] * len(data_loaders))
        self.store('client_model', client_model)
        self.store('data_loaders', data_loaders)

    def load_central_testset(self, model):
        dl = DataLoader(train_path=None, test_path=self.load('input_files')['central_test'][0])
        dl.lazy_init(train_batch_size=None,
                     test_batch_size=model.test_batch_size,
                     torch_mode=self.config['train_config']['torch_loader']
                     )
        dl = dl.test_loader
        self.store('test_loader', dl)


class LocalUpdate(AppState, ABC):
    """ Local Model training
        Input:
            Model weights(Coordinator already has it)
            App statuses: {Converged: True/False }
    """

    def run(self) -> str or None:
        client_model = self.load('client_model')
        data_loaders = self.load('data_loaders')
        state_dict = self.load('state_dict')
        self.update(message=f"#{self.load('iteration') + 1}: Waiting for Coordinator")
        weights, converged = self.get_global_parameters(client_model, n_splits=len(data_loaders))

        if all(converged):
            return 'Converged'

        weights, state_dict, data_loaders = \
            self.remove_converged_models(weights, state_dict, data_loaders, converged)
        data_to_send = self.local_computation(client_model, data_loaders, weights, state_dict)
        self.send_data_to_coordinator(data_to_send)

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

    def remove_converged_models(self, weights, state_dict, data_loaders, converged):
        if any(converged):
            weights = list(compress(weights, converged))
            state_dict = list(compress(state_dict, converged))
            data_loaders = list(compress(data_loaders, converged))
        return weights, state_dict, data_loaders

    def local_computation(self, client_model, data_loaders, weights, state_dicts):
        self.store('iteration', self.load('iteration') + 1)
        self.update(message=f"#{self.load('iteration')}: Local Training")
        new_parameters, new_state_dicts, trained_samples = [], [], []
        for counter, (dl, w, sd) in enumerate(zip(data_loaders, weights, state_dicts)):
            self.log(f"Iteration {self.load('iteration')}: Update model #{counter}")
            self.log(np.shape(w))
            client_model.update(dl.train_loader, w, sd, verbose=True)
            if dl.test_loader is not None:
                client_model.evaluate(dl.test_loader)
            new_parameters.append(client_model.get_weights())
            trained_samples.append(client_model.num_trained_samples)
            new_state_dicts.append(client_model.get_optimizer_params())
        self.store('state_dict', new_state_dicts)
        return list(zip(new_parameters, trained_samples))


class GlobalAggregation(AppState, ABC):
    def run(self) -> str or None:
        self.update(message=f"#{self.load('iteration')}: Waiting for others")
        received_params = self.gather_data()
        self.update(message=f"#{self.load('iteration')}: Aggregation")
        global_weights, stopping_criteria = self.global_aggregation(received_params)
        data_to_send = [global_weights, stopping_criteria]
        self.broadcast_data(data_to_send, send_to_self=False)
        if all(stopping_criteria):
            return "Converged"

    def global_aggregation(self, params):
        client_model = self.load('client_model')

        global_weights = self.average_weights(params, client_model)
        stopping_criteria = self.test_aggregated_models(global_weights, client_model)

        self.store('weights', global_weights)
        self.store('converged', stopping_criteria)
        return global_weights, stopping_criteria

    def average_weights(self, params, client_model):
        n_splits = len(self.load('data_loaders'))
        global_weights = [np.array(client_model.get_weights(), dtype='object') * 0] * n_splits
        total_n_samples = [0] * n_splits
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
        for counter, dl in enumerate(self.load('data_loaders')):
            if dl.test_loader is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                y_pred, y_true = client_model.predict(dl.test_loader)
                pd.DataFrame(y_pred, columns=['y_pred']).to_csv(self.load('output_files')['pred'][counter], index=None)
                pd.DataFrame(y_true, columns=['y_true']).to_csv(self.load('output_files')['target'][counter],
                                                                index=None)
        self.update(message="Finished!")

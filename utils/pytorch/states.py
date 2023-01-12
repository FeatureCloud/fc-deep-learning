import os
from abc import ABC
import numpy as np
from FeatureCloud.app.engine.app import AppState, LogLevel
from FeatureCloud.app.engine.app import State as op_state
from CustomStates import ConfigState
from utils.utils import design_model, get_dataloader, to_list, to_numpy, average_weights, \
    inject_root_path_to_clients_dir, get_path_to_central_test_output_files, remove_converged_models
from utils.pytorch.DeepModel import Model
from utils.pytorch.ClientModels import ClientModels
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
        self.initialize()
        self.store('config', self.config)
        if self.config.get('simulation', None) is not None:
            self.update(message="Simulation mode")
            self.log("The app will run in the simulation mode...")
            return ''
        dl = self.get_dataloader(self.load('input_files')['train'][0])
        data_cv_folds = zip(self.load('input_files')['train'], self.load('input_files')['test'])
        client_model, train_loaders, test_loaders = self.load_clients_data(data_cv_folds, dl)
        if self.is_coordinator and self.config['local_dataset']['central_test'] is not None:
            test_loader = dl.load(self.load('input_files')['central_test'][0], client_model.model.test_batch_size)
            self.store('test_loader', test_loader)
        self.store('n_splits', len(train_loaders))
        self.store('train_loaders', train_loaders)
        self.store('test_loaders', test_loaders)
        self.store('client_model', client_model)
        self.store('smpc_used', self.config.get('use_smpc', False))
        self.store('iteration', 0)
        self.store('config', self.config)
        if self.is_coordinator and len(self.clients) > 1:
            data_to_send = [self.load('client_model').get_weights()]
            self.broadcast_data(data=data_to_send, send_to_self=False)

    def initialize(self):
        self.update(state=op_state.RUNNING)
        self.update(message="Reading the config file....")
        self.read_config()
        self.lazy_init()
        self.finalize_config()

    def get_dataloader(self, sample_train_set):

        if self.load('input_files')['test'] is None and self.load('input_files')['central_test'] is None:
            self.log("There is no test data provided", LogLevel.ERROR)
            self.update(message="no test data", state=op_state.ERROR)
        self.log(f"Getting sample shape from {sample_train_set} dataset")

        dl_class = get_dataloader(self.config['train_config']['data_loader'])
        if dl_class is None:
            self.log(f"module {self.config['train_config']['data_loader']} neither is found in implemented models nor "
                     f"in `/mnt/input` directory", LogLevel.ERROR)
            self.update(message="module not found", state=op_state.ERROR)
        dl = dl_class(sample_train_set, **self.config['local_dataset']['detail'])
        return dl

    def build_client_model(self, data_loader, device):
        model_class, config = design_model(deepcopy(self.config['model']), data_loader)
        model = Model(model_class, config, self.config['train_config'], device)
        train_all = self.config["fed_hyper_params"]["federated_model"].strip().lower() == "fedavg"
        client_model = ClientModels(model, train_all, self.config["fed_hyper_params"]["batch_count"])
        # if self.coordinator:
        #     client_model = ClientModels(model, train_all,
        #                                 self.config["fed_hyper_params"]["batch_count"])
        # else:
        #     client_model = ClientModels(model, train_all,
        #                                 self.config["fed_hyper_params"]["batch_count"])
        return client_model

    def load_clients_data(self, data_cv_folds, dl, client_model=None):
        device = torch.device('cuda' if torch.cuda.is_available() and self.config['gpu'] else 'cpu')
        train_loaders, test_loaders = [], []
        for counter, (train_path, test_path) in enumerate(data_cv_folds):
            if client_model is None:
                client_model = self.build_client_model(dl.sample_data_loader, device)
            tr_dl = dl.load(train_path, client_model.model.batch_size)
            test_dl = dl.load(test_path, client_model.model.test_batch_size)
            train_loaders.append(tr_dl)
            test_loaders.append(test_dl)
        return client_model, train_loaders, test_loaders


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
        n_splits = len(train_loaders)
        self.state_dict = [client_model.get_optimizer_params()] * n_splits
        self.update(message=f"#{self.load('iteration') + 1}: Waiting for Coordinator")
        weights, converged = self.get_global_parameters(client_model, n_splits=self.load('n_splits'))

        if all(converged):
            return 'Converged'

        weights, state_dict, train_loaders, test_loaders = \
            remove_converged_models(weights, self.state_dict, train_loaders, test_loaders, converged)
        new_parameters, trained_samples, self.state_dict = self.local_computation(client_model, train_loaders,
                                                                                  test_loaders, weights,
                                                                                  self.state_dict)
        self.send_data_for_aggregation(list(zip(new_parameters, trained_samples)))

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

    def local_computation(self, client_model, train_loaders, test_loaders, weights, state_dicts):
        self.store('iteration', self.load('iteration') + 1)
        self.update(message=f"#{self.load('iteration')}: Local Training")
        new_parameters, new_state_dicts, trained_samples = [], [], []
        for counter, (tr_dl, test_dl, w, sd) in enumerate(zip(train_loaders, test_loaders, weights, state_dicts)):

            self.log(f"Iteration {self.load('iteration')}: Update model #{counter}")
            # TODO: testing models before local update and calculate the averaged performance
            # client_model.set_weights(w)
            # if test_dl is not None:
            #     client_model.evaluate(test_dl)
            client_model.update(tr_dl, w, sd, verbose=True)
            # TODO: test the local performance after local update and found the effect!
            if test_dl is not None:
                client_model.evaluate(test_dl)
            new_parameters.append(client_model.get_weights())
            trained_samples.append(client_model.num_trained_samples)
            new_state_dicts.append(client_model.get_optimizer_params())

        return new_parameters, trained_samples, new_state_dicts


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
            global_weights = average_weights(params)
        stopping_criteria = self.test_aggregated_models(global_weights, client_model)

        self.store('weights', global_weights)
        self.store('converged', stopping_criteria)
        return global_weights, stopping_criteria

    def test_aggregated_models(self, global_weights, client_model):
        iteration = self.load('iteration')
        max_iter = self.load('config')['fed_hyper_params']["max_iter"]
        test_set = self.load('test_loader')
        stopping_criteria = []
        for counter, w in enumerate(global_weights):
            self.update(message=f"Global aggregation")
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
        central_test_loader = self.load('test_loader')
        central_pred_file, central_target_file = get_path_to_central_test_output_files()
        test_loaders = self.load('test_loaders')
        pred_files = self.load('output_files')['pred']
        target_files = self.load('output_files')['target']
        # TODO: check updated weights are used here ==> client_model.set_weights()
        self.write_central_test_results(client_model, central_test_loader, central_pred_file, central_target_file)
        self.write_local_test_results(client_model, test_loaders, pred_files, target_files)
        self.update(message="Finished!")

    def write_central_test_results(self, client_model, test_loader, pred_file, target_file):
        if self.is_coordinator and test_loader is not None:
            self.log(f"Writing the results for centralized test")
            y_pred, y_true = client_model.predict(self.load('test_loader'))
            pd.DataFrame(y_pred, columns=['y_pred']).to_csv(pred_file, index=None)
            pd.DataFrame(y_true, columns=['y_true']).to_csv(target_file, index=None)

    def write_local_test_results(self, client_model, test_loaders, pred_files, target_files):
        for counter, (dl, pred_file, target_file) in enumerate(zip(test_loaders, pred_files, target_files)):
            if dl is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                y_pred, y_true = client_model.predict(dl)
                pd.DataFrame(y_pred, columns=['y_pred']).to_csv(pred_file, index=None)
                pd.DataFrame(y_true, columns=['y_true']).to_csv(target_file, index=None)


class Centralized(AppState, ABC):
    def run(self) -> str:
        central_test_loader = self.load('test_loader')
        client_model = self.load('client_model')
        train_loaders = self.load('train_loaders')
        test_loaders = self.load('test_loaders')
        n_splits = len(train_loaders)
        state_dict = [client_model.get_optimizer_params()] * n_splits
        weights = client_model.get_weights()
        client_model.model.epochs = self.load('config')['fed_hyper_params']["max_iter"]
        for counter, (tr_dl, test_dl, sd) in enumerate(zip(train_loaders, test_loaders, state_dict)):
            self.log(f"Training model #{counter}")
            client_model.set_weights(weights)
            client_model.set_optimizer_params(sd)
            client_model.model.fit(tr_dl, test_dl, verbose=True)
            if central_test_loader is not None:
                client_model.model.evaluate(central_test_loader)
        self.update(message="Finished!")


class Simulation(Initialization, LocalUpdate, GlobalAggregation, WriteResults, ABC):
    client_model = None
    weights = []
    n_splits = 0
    clients_data_loaders = []
    n_clients = 0
    max_iter = 0
    clients_state_dict = []
    central_test_loader = None
    global_weights = []
    batch_size = 0
    test_batch_size = 0
    data_loaders = []
    clients_dirs = []
    state_dicts = []

    def run(self) -> str:
        # self.initialize()
        self.config = self.load('config')
        self.correct_clients_dir()
        dl = self.get_dataloader(self.clients_input_files[0][0][0])
        self.load_clients_data(None, dl)
        if self.config['local_dataset']['central_test'] is not None:
            central_testset_path = inject_root_path_to_clients_dir(self.clients_dirs[0],
                                                                   self.load('input_files')['central_test']
                                                                   )[0]
            self.central_test_loader = dl.load(central_testset_path, self.client_model.model.test_batch_size)

        self.max_iter = self.load('config')['fed_hyper_params']['max_iter']
        self.global_weights = [self.client_model.get_weights()] * self.n_splits
        # print(self.client_model.get_optimizer_params())
        self.state_dicts = [[self.client_model.get_optimizer_params()] * self.n_splits] * self.n_clients
        for c_round in range(1, self.max_iter):
            self.log(f"#{c_round} communication round...")
            self.store('iteration', c_round)
            new_params = []
            for client in range(self.n_clients):
                self.log(f"Local training for client #{client}")
                new_parameters, n_trained_samples, self.state_dicts[client] = \
                    self.local_computation(self.client_model,
                                           self.data_loaders[client]['train_loaders'],
                                           self.data_loaders[client]['test_loaders'],
                                           self.global_weights,
                                           self.state_dicts[client]
                                           )
                self.log(n_trained_samples)
                print(np.shape(new_parameters[0]))
                new_params.append(list(zip(new_parameters, n_trained_samples)))
            average_weights(new_params)
            stopping_criteria = self.test_aggregated_models(self.global_weights, self.client_model)
            if stopping_criteria:
                break
        self.write_results()

    def write_results(self):
        if self.config['local_dataset']['central_test'] is not None:
            # only the coordinator, i.e., first client, will receive the central testset
            pred_file, target_file = get_path_to_central_test_output_files()
            pred_file = inject_root_path_to_clients_dir(self.clients_dirs[0], [pred_file], input=False)[0]
            target_file = inject_root_path_to_clients_dir(self.clients_dirs[0], [target_file], input=False)[0]
            self.write_central_test_results(self.client_model, self.central_test_loader, pred_file, target_file)

        for client, dir in enumerate(self.clients_dirs):
            os.mkdir(f"/mnt/output/{dir}")
            pred_files = inject_root_path_to_clients_dir(dir, self.load('output_files')['pred'], input=False)
            target_files = inject_root_path_to_clients_dir(dir, self.load('output_files')['target'], input=False)
            self.write_local_test_results(self.client_model,
                                          self.data_loaders[client]['test_loaders'],
                                          pred_files,
                                          target_files)

    def load_clients_data(self, data_cv_folds, dl, client_model=None):
        self.data_loaders = []
        for client_files in self.clients_input_files:
            self.client_model, train_loaders, test_loaders = \
                super(Simulation, self).load_clients_data(zip(*client_files), dl, self.client_model)
            self.data_loaders.append({'train_loaders': train_loaders, 'test_loaders': test_loaders})
        self.n_splits = len(train_loaders)

    def correct_clients_dir(self):
        self.clients_dirs = self.config['simulation']['clients_dir'].split(",")
        self.n_clients = len(self.clients_dirs)
        self.clients_input_files = []
        for client_dir in self.clients_dirs:
            train_files = inject_root_path_to_clients_dir(client_dir, self.load('input_files')['train'])
            test_files = inject_root_path_to_clients_dir(client_dir, self.load('input_files')['test'])
            self.clients_input_files.append([train_files, test_files])
import copy
import os
from abc import ABC
import numpy as np
from FeatureCloud.app.engine.app import AppState, LogLevel
from FeatureCloud.app.engine.app import State as op_state
from CustomStates import ConfigState
from utils import utils
from utils.pytorch.ClientModels import ClientModels
from utils.pytorch.utils import GlobalUpdates, LocalUpdates
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
        if self.config.get('simulation', None) is not None:
            self.update(message="Simulation mode")
            self.log("The app will run in the simulation mode...")
            return 'simulation'

        dl = self.get_dataloader(self.load('input_files')['train'][0])
        self.get_custom_modules()
        data_cv_folds = zip(self.load('input_files')['train'], self.load('input_files')['test'])
        client_model, train_loaders, test_loaders = self.load_clients_data(data_cv_folds, dl)
        test_loader = None
        if self.is_coordinator:
            if self.config['local_dataset'].get('central_test', None) is not None:
                test_loader = dl.load(self.load('input_files')['central_test'][0], client_model.model.test_batch_size)
                # self.store('test_loader', test_loader)
            if self.config['local_dataset'].get('init_model', None) is not None:
                client_model.load_model(self.load('input_files')['init_model'][0])

        self.store_(test_loader, train_loaders, test_loaders, client_model)

        if self.config.get('centralized', None) is not None:
            self.update(message="Centralized training")
            self.log("The app will run in the Centralized mode...")
            return 'centralized'

        if self.is_coordinator:
            data_to_send = [self.load('client_model').get_weights()]
            self.broadcast_data(data=data_to_send)

    def store_(self, test_loader, train_loaders, test_loaders, client_model):
        self.load_schemas()

        self.store('test_loader', test_loader)
        self.store('n_splits', len(train_loaders))
        self.store('train_loaders', train_loaders)
        self.store('test_loaders', test_loaders)
        self.store('client_model', client_model)
        self.store('smpc_used', self.config.get('use_smpc', False))
        self.store('iteration', 0)
        self.store('config', self.config)

    def load_schemas(self):
        g = getattr(GlobalUpdates, self.config["fed_hyper_params"]['global_updates']).value
        l = getattr(LocalUpdates, self.config["trainer"]['local_updates']).value
        self.store('global_update_schema', g)
        self.store('local_update_schema', l)

    def get_custom_modules(self):
        self.trainer_config = {}
        modules = {}

        # Metrics
        metrics = []
        for metric in self.config['trainer']['metrics']:
            m = {'name': metric['name'],
                 'func': utils.get_metrics(metric['name'], metric['package']),
                 'param': metric.get('param', {})}
            metrics.append(m)
        modules['metrics'] = metrics

        self.trainer_config['metrics'] = metrics

        # Loss function
        l = utils.get_loss_func(self.config['trainer']['loss']['name'])
        self.trainer_config['loss'] = {'func': l, 'param': self.config['trainer']['loss'].get('param', {})}

        # Optimizer
        optimizer = utils.get_optimizer(self.config['trainer']['optimizer']['name'])
        self.trainer_config['optimizer'] = {'opt': optimizer,
                                            'param': self.config['trainer']['optimizer'].get('param', {})
                                            }

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

        dl_class = utils.get_dataloader(self.config['trainer']['data_loader'])
        if dl_class is None:
            self.log(f"module {self.config['trainer']['data_loader']} neither is found in implemented models nor "
                     f"in `{utils.get_root_path()}` directory", LogLevel.ERROR)
            self.update(message="module not found", state=op_state.ERROR)
        dl = dl_class(sample_train_set, **self.config['local_dataset']['detail'])
        return dl

    def build_client_model(self, data_loader):
        self.config['train_config']['device'] = utils.set_device(self.config['train_config']['device'])
        dnn_architecture, dnn_config = utils.design_architecture(deepcopy(self.config['model']), data_loader)
        local_trainer = utils.get_trainer(self.config['trainer']['name'])
        model = local_trainer(model=dnn_architecture,
                              config=dnn_config,
                              attributes=self.trainer_config,
                              req_local_updates=self.load('local_update_schema'),
                              log=self.log,
                              train_config=self.config['train_config'],
                              **self.config['trainer'].get('param', {}))

        aggregator = None
        # Aggregator
        if self.is_coordinator:
            aggregator_class = utils.get_aggregator(self.config["fed_hyper_params"]["federated_model"])
            param = self.config["fed_hyper_params"]
            param.update({'train_config': self.config['train_config']})
            aggregator = aggregator_class(**param)
        client_model = ClientModels(model, aggregator)

        return client_model

    def load_clients_data(self, data_cv_folds, dl, client_model=None):

        train_loaders, test_loaders = [], []
        for counter, (train_path, test_path) in enumerate(data_cv_folds):
            if client_model is None:
                client_model = self.build_client_model(dl.sample_data_loader)
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
        iteration = self.load('iteration')
        n_splits = len(train_loaders)
        self.backup = [client_model.local_backup()] * n_splits
        self.update(message=f"#{self.load('iteration')}: Waiting for Coordinator")
        if self.is_coordinator and iteration > 0:
            received_data = self.load('received_data')
        else:
            received_data = self.await_data(unwrap=True)
        global_update_dict = self.preprocess_global_updates(iteration, received_data)

        if all(global_update_dict['stoppage']):
            self.store('weights', global_update_dict['weights'])
            return 'Converged'

        global_update_dict, self.backup, train_loaders, test_loaders = \
            utils.remove_converged_models(global_update_dict, self.backup, train_loaders, test_loaders)

        data_to_send = self.local_computation(client_model, train_loaders, test_loaders, global_update_dict, iteration)
        self.send_data_for_aggregation(data_to_send)
        self.store('iteration', iteration + 1)

    def preprocess_global_updates(self, iteration, received_data):
        if iteration == 0:
            received_data = [received_data] * self.load('n_splits')
            received_data.append([False] * len(received_data[0]))
        utils.interpret_global_updates(received_data, self.load('global_update_schema'))
        global_update_dict = utils.unpack(received_data, self.load('global_update_schema'))
        return global_update_dict

    def send_data_for_aggregation(self, data):
        data_to_send = data
        if self.load('smpc_used'):
            data_to_send = utils.to_list([[np.array(d, dtype='object') * w, w] for d, w in data])
        self.send_data_to_coordinator(data_to_send, use_smpc=self.load('smpc_used'))

    def local_computation(self, client_model, train_loaders, test_loaders, global_updates, iteration):
        self.update(message=f"#{iteration}: Local Training")
        data_to_send, local_backup = [], []
        cv_first_updates = utils.cv_first(global_updates)
        for counter, (tr_dl, test_dl, updates, backup) in enumerate(
                zip(train_loaders, test_loaders, cv_first_updates, self.backup)):
            self.log(f"Iteration {iteration}: Update model #{counter}")
            self.pre_update_eval(client_model, test_dl)
            client_model.update(tr_dl, updates, backup, test_dl)
            data_to_send.append(client_model.get_local_updates(self.load('local_update_schema')))
            self.post_update_eval(client_model, test_dl)
            local_backup.append(client_model.local_backup())
        self.backup = local_backup
        return data_to_send

    def pre_update_eval(self, client_model, dl):
        if dl is not None:
            self.log("Local model evaluation BEFORE local update")
            client_model.evaluate(dl)

    def post_update_eval(self, client_model, dl):
        if dl is not None:
            self.log("Local model evaluation AFTER local update")
            client_model.evaluate(dl)


class GlobalAggregation(AppState, ABC):
    def run(self) -> str or None:
        self.update(message=f"#{self.load('iteration')}: Waiting for others")
        received_params = self.gather_local_models()
        self.update(message=f"#{self.load('iteration')}: Aggregation")
        data_to_send = self.aggregate(received_params, self.load('client_model'), self.load('test_loader'))

        self.store('received_data', data_to_send)

        self.broadcast_data(data_to_send, send_to_self=False)
        global_update_dict, converged = self.all_converged(data_to_send)
        if converged:
            self.store('weights', global_update_dict['weights'])
            return "Converged"

    def all_converged(self, data_to_send):
        utils.interpret_global_updates(data_to_send, self.load('global_update_schema'))
        global_update_dict = utils.unpack(data_to_send, self.load('global_update_schema'))
        converged = all(global_update_dict['stoppage'])
        return global_update_dict, converged

    def aggregate(self, received_params, client_model, test_loader):
        aggregator = client_model.aggregator
        if self.load('smpc_used'):
            aggregator.aggregate_smpc(received_params)
        else:
            aggregator.aggregate(received_params)
        metrics = self.evaluate_aggregated_models(aggregator.weights,
                                                  client_model,
                                                  test_loader
                                                  )
        aggregator.post_aggregate(metrics=metrics)
        data_to_send = aggregator.get_global_updates()
        return data_to_send

    def evaluate_aggregated_models(self, global_weights, client_model, test_set):
        metrics = []
        for counter, w in enumerate(global_weights):
            self.update(message=f"Global aggregation")
            if test_set is not None:
                self.update(message=f"#{self.load('iteration')}: Test G model")
                client_model.set_weights(w)
                m = client_model.evaluate(test_set)
                metrics.append(m)
                self.log(f"Iteration #{self.load('iteration')}: Testing Global model on global test set #{counter}: {m}")
        return metrics

    def gather_local_models(self):
        if self.load('smpc_used'):
            return self.await_data(is_json=True)
        return self.gather_data()


class WriteResults(AppState, ABC):
    def run(self) -> str or None:
        self.config = self.load('config')
        self.update(message=f"Writing Results")
        client_model = self.load('client_model')
        central_test_loader = self.load('test_loader')
        central_pred_file, central_target_file = utils.get_path_to_central_test_output_files()
        test_loaders = self.load('test_loaders')
        pred_files = self.load('output_files')['pred']
        target_files = self.load('output_files')['target']
        # TODO: check updated weights are used here ==> client_model.set_weights()
        self.write_central_test_results(client_model, central_test_loader, central_pred_file, central_target_file,
                                        self.load('weights'))
        self.write_local_test_results(client_model, test_loaders, pred_files, target_files, self.load('weights'))
        self.write_dnn_models(client_model, self.load('output_files')['model'], self.load('weights'))
        self.update(message="Finished!")

    def write_central_test_results(self, client_model, test_loader, pred_file, target_file, weights):
        if self.is_coordinator and test_loader is not None:
            self.log(f"Writing the results for centralized test")
            client_model.set_weights(weights[0])
            y_pred, y_true = client_model.predict(self.load('test_loader'))
            pd.DataFrame(y_pred, columns=['y_pred']).to_csv(pred_file, index=None)
            pd.DataFrame(y_true, columns=['y_true']).to_csv(target_file, index=None)

    def write_local_test_results(self, client_model, test_loaders, pred_files, target_files, weights):
        for counter, (dl, w, pred_file, target_file) in enumerate(zip(test_loaders, weights, pred_files, target_files)):
            if dl is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                client_model.set_weights(w)
                y_pred, y_true = client_model.predict(dl)
                pd.DataFrame(y_pred, columns=['y_pred']).to_csv(pred_file, index=None)
                pd.DataFrame(y_true, columns=['y_true']).to_csv(target_file, index=None)

    def write_dnn_models(self, client_model, dirs, weights):
        if self.config['result'].get('model', None) is not None:
            for counter, (model_file, weight) in enumerate(zip(dirs, weights)):
                client_model.set_weights(weight)
                client_model.store_model(model_file)


class Centralized(LocalUpdate, GlobalAggregation, WriteResults, ABC):
    """
    All the initializations steps were ran before entering centralized
    """

    def run(self) -> str:
        self.config = self.load('config')
        central_test_loader = self.load('test_loader')
        client_model = self.load('client_model')
        train_loaders = self.load('train_loaders')
        test_loaders = self.load('test_loaders')
        central_pred_file, central_target_file = utils.get_path_to_central_test_output_files()
        n_splits = len(train_loaders)
        state_dict = [client_model.get_optimizer_params()] * n_splits
        updates = {'weights': client_model.get_weights(), 'config': None}
        client_model.model.epochs = self.load('config')['fed_hyper_params']["max_iter"]
        trained_models = []
        for counter, (tr_dl, test_dl, sd) in enumerate(zip(train_loaders, test_loaders, state_dict)):
            self.log(f"Training model #{counter}")
            client_model.update(data_loader=tr_dl,
                                global_updates=updates,
                                backup=sd,
                                test_loader=test_dl)
            trained_models.append(client_model.get_weights())

        self.write_central_test_results(client_model, central_test_loader, central_pred_file, central_target_file,
                                        trained_models)
        self.write_local_test_results(client_model, test_loaders, self.load('output_files')['pred'],
                                      self.load('output_files')['target'], trained_models)
        self.write_dnn_models(client_model, self.load('output_files')['model'], trained_models)
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
    backup = []

    def run(self) -> str:
        # self.initialize()
        self.initial_operations()
        clients_backup = [[self.client_model.local_backup()] * self.n_splits] * self.n_clients
        for c_round in range(self.max_iter):
            self.log(f"Communication round: #{c_round + 1} ...")
            gathered_data = []

            ### LocalUpdate
            for client in range(self.n_clients):
                self.log(f"Client #{client + 1}: Local training")
                self.backup = clients_backup[client]
                global_update_dict = self.preprocess_global_updates(c_round, self.received_data)
                if all(global_update_dict['stoppage']):
                    break
                global_update_dict, \
                self.backup, \
                self.data_loaders[client]['train_loaders'], \
                self.data_loaders[client]['train_loaders'] = \
                    utils.remove_converged_models(
                        global_update_dict,
                        self.backup,
                        self.data_loaders[client]['train_loaders'],
                        self.data_loaders[client]['train_loaders']
                    )
                data_to_send = \
                    self.local_computation(self.client_model,
                                           self.data_loaders[client]['train_loaders'],
                                           self.data_loaders[client]['test_loaders'],
                                           global_update_dict,
                                           c_round + 1
                                           )
                clients_backup[client] = self.backup
                gathered_data.append(data_to_send)

            ### GlobalAggregation
            self.received_data = self.aggregate(gathered_data, self.client_model, self.central_test_loader)

            global_update_dict, converged = self.all_converged(self.received_data)
            if converged:
                break

        ### WriteResults

        self.write_central_results(global_update_dict['weights'])
        self.write_dnn_models(self.client_model, self.load('output_files')['model'], global_update_dict['weights'])

    def initial_operations(self):
        self.initialize()
        self.correct_clients_dir()
        self.get_custom_modules()
        dl = self.get_dataloader(self.clients_input_files[0][0][0])
        self.load_clients_data(None, dl)
        if self.config['local_dataset']['central_test'] is not None:
            central_testset_path = utils.inject_root_path_to_clients_dir(self.clients_dirs[0],
                                                                         self.load('input_files')['central_test']
                                                                         )[0]
            self.central_test_loader = dl.load(central_testset_path, self.client_model.model.test_batch_size)
        self.max_iter = self.config['fed_hyper_params']['max_iter']
        self.received_data = [self.client_model.get_weights()] * self.n_splits
        self.store('n_splits', self.n_splits)
        self.load_schemas()

    def write_central_results(self, weights):
        if self.config['local_dataset']['central_test'] is not None:
            # only the coordinator, i.e., first client, will receive the central testset
            pred_file, target_file = utils.get_path_to_central_test_output_files()
            pred_file = utils.inject_root_path_to_clients_dir(self.clients_dirs[0], [pred_file], input=False)[0]
            target_file = utils.inject_root_path_to_clients_dir(self.clients_dirs[0], [target_file], input=False)[0]
            self.write_central_test_results(self.client_model, self.central_test_loader, pred_file, target_file,
                                            weights)

        for client, dir in enumerate(self.clients_dirs):
            path = f"{utils.get_root_path(input=False)}/{dir}"
            if not os.path.exists(path):
                os.mkdir(path)
            pred_files = utils.inject_root_path_to_clients_dir(dir, self.load('output_files')['pred'], input=False)
            target_files = utils.inject_root_path_to_clients_dir(dir, self.load('output_files')['target'], input=False)
            self.write_local_test_results(self.client_model,
                                          self.data_loaders[client]['test_loaders'],
                                          pred_files,
                                          target_files,
                                          self.global_weights)

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
            train_files = utils.inject_root_path_to_clients_dir(client_dir, self.load('input_files')['train'])
            test_files = utils.inject_root_path_to_clients_dir(client_dir, self.load('input_files')['test'])
            self.clients_input_files.append([train_files, test_files])

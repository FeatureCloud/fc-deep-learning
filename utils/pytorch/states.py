import os
from abc import ABC
import numpy as np
from FeatureCloud.app.engine.app import AppState, LogLevel
from FeatureCloud.app.engine.app import State as op_state
from CustomStates import ConfigState
from utils import utils
from utils.pytorch.utils import LocalUpdates, GlobalUpdates, write_preds, TensorBoardWriter
from utils.pytorch.ClientModels import ClientModels
import pandas as pd
from copy import deepcopy
import copy


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExtendedConfigState(ConfigState.State, ABC):
    """
    Read input data
    read config files

    """

    def get_custom_modules(self):
        self.trainer_config = {}
        modules = {}

        # Metrics
        metrics = []
        for metric in self.config['trainer']['metrics']:
            m = {'name': metric['name'],
                 'func': utils.get_metrics(metric['name'], metric['package'], self.input_dir),
                 'param': metric.get('param', {})}
            metrics.append(m)
        modules['metrics'] = metrics

        self.trainer_config['metrics'] = metrics

        # Loss function
        l = utils.get_loss_func(self.config['trainer']['loss']['name'], self.input_dir)
        self.trainer_config['loss'] = {'func': l, 'param': self.config['trainer']['loss'].get('param', {})}

        # Optimizer
        optimizer = utils.get_optimizer(self.config['trainer']['optimizer']['name'], self.input_dir)
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

        dl_class = utils.get_dataloader(self.config['trainer']['data_loader'], self.input_dir)
        if dl_class is None:
            self.log(f"module {self.config['trainer']['data_loader']} neither is found in implemented models nor "
                     f"in `{utils.get_root_path()}` directory", LogLevel.ERROR)
            self.update(message="module not found", state=op_state.ERROR)
        dl = dl_class(sample_train_set, **self.config['local_dataset']['detail'])
        return dl

    def build_client_model(self, data_loader):
        gpu = self.config['train_config']['device'].strip().lower() == "gpu"
        self.config['train_config']['device'] = utils.set_device(self.device, gpu)
        dnn_architecture, dnn_config = utils.design_architecture(deepcopy(self.config['model']), data_loader,
                                                                 self.input_dir)
        local_trainer = utils.get_trainer(self.config['trainer']['name'], self.input_dir)
        model = local_trainer(model=dnn_architecture,
                              config=dnn_config,
                              attributes=self.trainer_config,
                              req_local_updates=self.load('local_update_schema'),
                              log=self.log,
                              train_config=copy.deepcopy(self.config['train_config']),
                              **self.config['trainer'].get('param', {}))

        aggregator = None
        # Aggregator
        if self.is_coordinator:
            aggregator_class = utils.get_aggregator(self.config["fed_hyper_params"]["federated_model"], self.input_dir)
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

    def load_schemas(self):
        g = getattr(GlobalUpdates, self.config["fed_hyper_params"]['global_updates']).value
        l = getattr(LocalUpdates, self.config["trainer"]['local_updates']).value
        self.store('global_update_schema', g)
        self.store('local_update_schema', l)


#
#
class Initialization(ExtendedConfigState, ABC):
    """
    Read input data
    read config files

    """

    def run(self) -> str or None:
        self.initialize()
        dl = self.get_dataloader(self.load('input_files')['train'][0])
        self.get_custom_modules()
        metrics = [m["name"] for m in self.trainer_config['metrics']]
        data_cv_folds = zip(self.load('input_files')['train'], self.load('input_files')['test'])
        client_model, train_loaders, test_loaders = self.load_clients_data(data_cv_folds, dl)
        models = [f"Model{i}" for i in range(len(train_loaders))]
        logdir = f"{self.output_dir}/TsBoard_logs"
        tsboard_writer = TensorBoardWriter(logdir, [self.id], models, metrics)
        self.store("tsboard_writer", tsboard_writer)
        test_loader = None
        if self.is_coordinator:
            if self.config['local_dataset'].get('central_test', None) is not None:
                test_loader = dl.load(self.load('input_files')['central_test'][0], client_model.model.test_batch_size)
                # self.store('test_loader', test_loader)
            if self.config['local_dataset'].get('init_model', None) is not None:
                client_model.load_model(self.load('input_files')['init_model'][0])

        self.store_(test_loader, train_loaders, test_loaders, client_model)

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
        self.store("output_dir", self.output_dir)


class LocalUpdate(AppState, ABC):
    """ Local Model training
        Input:
            Model weights(Coordinator already has it)
            App statuses: {Converged: True/False }
    """

    def run(self) -> str or None:
        self.tsboard_writer = self.load("tsboard_writer")
        client_model = self.load('client_model')
        client_model.model.log = self.log
        train_loaders = self.load('train_loaders')
        test_loaders = self.load('test_loaders')
        iteration = self.load('iteration')
        n_splits = len(train_loaders)
        self.backup = [client_model.local_backup()] * n_splits
        self.update(message=f"#{self.load('iteration') + 1}: Waiting for Coordinator")
        if self.is_coordinator and iteration > 0:
            received_data = self.load('received_data')
        else:
            received_data = self.await_data(unwrap=True)
        global_update_dict = self.preprocess_global_updates(iteration, received_data)

        if all(global_update_dict['stoppage']):
            # print(np.shape(global_update_dict['weights']))
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
            self.pre_update_eval(client_model, test_dl, counter)
            tsboard = {"writer": self.tsboard_writer, "id": self.id, "model": f"Model{counter}"}
            client_model.update(tr_dl, updates, backup, test_dl, tsboard)
            data_to_send.append(client_model.get_local_updates(self.load('local_update_schema')))
            self.post_update_eval(client_model, test_dl, counter)
            local_backup.append(client_model.local_backup())
        self.backup = local_backup
        return data_to_send

    def pre_update_eval(self, client_model, dl, model_n):
        if dl is not None:
            self.log("Local model evaluation BEFORE local update")
            metrics = client_model.evaluate(dl)
            self.tsboard_writer.update(self.id, f"Model{model_n}", metrics, state="PreUpdateLocalTest")
            self.tsboard_writer.write_summaries(self.load("iteration"))

    def post_update_eval(self, client_model, dl, counter):
        if dl is not None:
            self.log("Local model evaluation AFTER local update")
            metrics = client_model.evaluate(dl)
            self.tsboard_writer.update(self.id, f"Model{counter}", metrics, state="PostUpdateLocalTest")
            self.tsboard_writer.write_summaries(self.load("iteration"))


class GlobalAggregation(AppState, ABC):
    def run(self) -> str or None:
        self.client_model = self.load('client_model')
        self.client_model.model.log = self.log
        self.update(message=f"#{self.load('iteration')}: Waiting for others")
        received_params = self.gather_local_models()
        self.update(message=f"#{self.load('iteration')}: Aggregation")
        data_to_send = self.aggregate(received_params, self.client_model, self.load('test_loader'))

        self.store('received_data', data_to_send)
        # coordinator has the weights in client_model
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
        aggregator.iteration += 1
        if self.load('smpc_used'):
            aggregator.aggregate_smpc(received_params, len(self.clients))
        else:
            aggregator.aggregate(received_params)
        metrics = self.evaluate_aggregated_models(aggregator.weights,
                                                  client_model,
                                                  test_loader
                                                  )
        self.log(metrics)
        aggregator.post_aggregate(metrics=metrics)
        data_to_send = aggregator.get_global_updates()
        return data_to_send

    def evaluate_aggregated_models(self, global_weights, client_model, test_set):
        metrics = []
        for counter, w in enumerate(global_weights):
            self.update(message=f"Global aggregation")
            if test_set is not None:
                self.update(message=f"#{self.load('iteration')}: Test G model")
                self.log(f"Iteration #{self.load('iteration')}: Testing Global model #{counter}")
                client_model.set_weights(w)
                metrics = client_model.evaluate(test_set)
                tsboard_writer = self.load("tsboard_writer")
                tsboard_writer.update(self.id, f"Model{counter}", metrics, state="GlobalTest")
                tsboard_writer.write_summaries(self.load("iteration"))
        return metrics

    def gather_local_models(self):
        if self.load('smpc_used'):
            return self.await_data(is_json=True)
        return self.gather_data()


class WriteResults(AppState, ABC):
    def run(self) -> str or None:
        self.load("tsboard_writer").close()
        self.config = self.load('config')
        self.update(message=f"Writing Results")
        client_model = self.load('client_model')
        client_model.model.log = self.log
        central_test_loader = self.load('test_loader')
        # central_pred_file, central_target_file = utils.get_path_to_central_test_output_files(self.load('output_dir'))
        test_loaders = self.load('test_loaders')
        pred_files = self.load('output_files')['pred']
        target_files = self.load('output_files')['target']
        # TODO: check updated weights are used here ==> client_model.set_weights()

        # write_central_test_results
        if self.is_coordinator and central_test_loader is not None:
            # Same test-set but different splits and weights
            for pred_file, target_file, w in zip(self.load('output_files')['central_pred'],
                                                 self.load('output_files')['central_target'],
                                                 self.load('weights')):
                client_model.set_weights(w)
                self.log(f"Writing the results for centralized test")
                write_preds(client_model, central_test_loader, pred_file, target_file)

        # write_local_test_results
        for counter, (dl, w, pred_file, target_file) in enumerate(zip(test_loaders,
                                                                      self.load('weights'),
                                                                      pred_files, target_files)
                                                                  ):
            if dl is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                client_model.set_weights(w)
                write_preds(client_model, dl, pred_file, target_file)

        # write_dnn_models
        if self.config['result'].get('model', False):
            for counter, (model_file, weight) in enumerate(
                    zip(self.load('output_files')['model'], self.load('weights'))):
                client_model.set_weights(weight)
                client_model.store_model(model_file)

        self.update(message="Finished!")


class Centralized(ExtendedConfigState, LocalUpdate, GlobalAggregation, ABC):
    """
    All the initializations steps were ran before entering centralized
    """

    # ConfigState.__init__(self, **kwargs)

    def run(self) -> str:
        self.initialize()
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

        central_test_loader = self.load('test_loader')
        central_pred_file, central_target_file = utils.get_path_to_central_test_output_files()
        n_splits = len(train_loaders)
        state_dict = [client_model.get_optimizer_params()] * n_splits
        updates = {'weights': client_model.get_weights(), 'config': None}
        trained_models = []
        for counter, (tr_dl, test_dl, sd) in enumerate(zip(train_loaders, test_loaders, state_dict)):
            self.log(f"Training model #{counter}")
            client_model.update(data_loader=tr_dl,
                                global_updates=updates,
                                backup=sd,
                                test_loader=test_dl)
            trained_models.append(client_model.get_weights())

        # write_central_test_results
        if self.is_coordinator and test_loader is not None:
            client_model.set_weights(trained_models)
            self.log(f"Writing the results for centralized test")
            write_preds(client_model, central_test_loader, central_pred_file, central_target_file)

        # write_local_test_results
        for counter, (dl, w, pred_file, target_file) in enumerate(zip(test_loaders,
                                                                      trained_models,
                                                                      self.load('output_files')['pred'],
                                                                      self.load('output_files')['target'])
                                                                  ):
            if dl is not None:
                self.log(f"Writing the results for of local test-set #{counter}")
                client_model.set_weights(w)
                write_preds(client_model, dl, pred_file, target_file)

        # write_dnn_models
        if self.config['result'].get('model', False):
            for counter, (model_file, weight) in enumerate(zip(self.load('output_files')['model'], trained_models)):
                client_model.set_weights(weight)
                client_model.store_model(model_file)

        self.update(message="Finished!")

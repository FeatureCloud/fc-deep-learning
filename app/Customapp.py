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
from .logic import bcolors
from .Customlogic import CustomLogic
from app.DeepLearning.utils import load_data_loader
from app.DeepLearning.ClientModels import ClientModels
from app.DeepLearning.utils import design_model
from app.DeepLearning.Baseline.DeepModel import Model
from app.DeepLearning.DataLoader import DataLoader
import numpy as np
import bios
import os
import pandas as pd
import shutil
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import sys
import gc


class DeepApp(CustomLogic):
    """
    Attributes
    ----------
    train_filename: str
    test_filename: str
    pred_filename: str
    y_true_filename: str
    federated_hyper_parameters: dict
    train_config: dict
    model_config: dict
    torch_loader: dict
    model: DeepModel.Model
    client_model: ClientModels
    data_loaders: dict
    state_dict: dict

    Methods
    -------
     read_config(config_file)
     finalize_config()
     read_input()
     broadcasting_init_params()
     local_computation()
     global_aggregation()
     write_results()
    """

    def __init__(self):
        super(DeepApp, self).__init__()

        self.train_filename = None
        self.test_filename = None
        self.pred_filename = None
        self.y_true_filename = None

        #   Configs
        self.federated_hyper_parameters = {}
        self.train_config = {}
        self.model_config = {}
        self.torch_loader = None

        #   Models & Parameters
        self.model = None
        self.client_model = None
        self.data_loaders = {}
        self.state_dict = {}

        #  Update States Functionality

        self.states["Read Input"] = self.read_input
        self.states["Broadcasting Initial parameters"] = self.broadcasting_init_params
        self.states["Local Update"] = self.local_computation
        self.states["Global Aggregation"] = self.global_aggregation
        self.states["Writing Results"] = self.write_results

    def read_config(self, config_file):
        """ should be overridden!
            reads the config file.
            calls the lazy_initialization method!

        Parameters
        ----------
        config_file: string
            path to the config.yaml file!

        """
        config = bios.read(config_file)['fc_deep']
        self.train_filename = config['local_dataset']['train']
        self.test_filename = config['local_dataset']['test']
        self.pred_filename = config['results']['pred']
        self.y_true_filename = config["results"]["target"]
        self.train_config = config['train_config']
        self.torch_loader = self.train_config.pop('torch_loader')
        self.federated_hyper_parameters = config["federated_hyper_parameters"]
        self.model_config = config['model']
        self.lazy_initialization(**config["logic"])

    def finalize_config(self):
        """

        Returns
        -------

        """
        if self.mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
            self.state_dict = dict.fromkeys(self.splits.keys())
            self.parameters = dict.fromkeys(self.splits.keys())
            self.workflows_states = dict.fromkeys(self.splits.keys())
        else:
            self.splits[self.INPUT_DIR] = None
            self.state_dict[self.INPUT_DIR] = None
            self.parameters[self.INPUT_DIR] = None
            self.workflows_states[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        print(f'Read config file.', flush=True)

    def read_input(self):
        self.progress = "Config..."
        self.read_config(self.INPUT_DIR + '/config.yml')
        self.finalize_config()
        for split in self.splits.keys():
            print(f'{bcolors.SPLIT}Read {split}{bcolors.ENDC}')
            train_path = split + "/" + self.train_filename
            test_path = split + "/" + str(self.test_filename)
            self.data_loaders[split] = DataLoader(train_path, test_path)
            model_class, config = design_model(self.model_config,
                                               self.data_loaders[split].sample_data)
            train_all = True if self.federated_hyper_parameters[
                                    "federated_model"].strip().lower() == "fedavg" else False
            if self.model is None:
                self.model = Model(model_class, config, self.train_config)
            self.data_loaders[split].lazy_init(self.model.batch_size, self.model.test_batch_size, self.torch_loader)
            if self.client_model is None:
                if self.coordinator:
                    self.client_model = ClientModels(self.model, train_all,
                                                     self.federated_hyper_parameters["batch_count"])
                else:
                    self.client_model = ClientModels(self.model, train_all,
                                                     self.federated_hyper_parameters["batch_count"])
            self.state_dict[split] = self.client_model.get_optimizer_params()
        # self.sizeof()
        super(DeepApp, self).read_input()

    def broadcasting_init_params(self):
        self.progress = 'Preprocess...'
        for split in self.splits.keys():
            self.parameters[split] = self.client_model.get_weights()
            print(f"#{split}:{len(self.parameters[split])}")

        super(DeepApp, self).broadcasting_init_params()

    def local_computation(self):
        self.progress = 'local update'
        self.iteration += 1
        print(f'{bcolors.STATE} Iteration {self.iteration} {bcolors.ENDC}')
        data_to_send = {}
        for split in self.splits.keys():
            print(f'{bcolors.SPLIT} Compute {split} {bcolors.ENDC}')
            self.client_model.update(self.data_loaders[split].train_loader,
                                     self.parameters[split],
                                     self.state_dict[split],
                                     verbose=True)
            self.parameters[split] = [self.client_model.get_weights(), self.client_model.num_trained_samples]
            self.state_dict[split] = self.client_model.get_optimizer_params()

        gc.collect()
        super(DeepApp, self).local_computation()

    def global_aggregation(self):
        restricted_memory = True
        self.progress = 'Global Aggregation...'
        if len(self.data_incoming) == len(self.clients):
            print(f"{bcolors.SEND_RECEIVE} Received parameters of all clients. {bcolors.ENDC}")
            data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
            self.data_incoming = []
            for split in self.splits.keys():
                if not self.workflows_states[split]:
                    print(f'{bcolors.SPLIT} Aggregate {split} {bcolors.ENDC}')
                    clients_parameters = [client[split] for client in data]
                    global_weights = np.array(self.client_model.get_weights(), dtype='object') * 0
                    total = 0
                    for client in clients_parameters:
                        param, n_samples = client[0], client[1]
                        global_weights += np.array(param, dtype='object') * n_samples
                        total += n_samples
                    global_weights /= total
                    self.client_model.set_weights(global_weights)
                    loss, acc = self.client_model.evaluate(self.data_loaders[split].train_loader_for_test)
                    print(f"{bcolors.VALUE} Iteration[{self.iteration}]"
                          f"\tTest Accuracy: {acc:.2f}"
                          f"\tTest Loss: {loss:.2f} {bcolors.ENDC}")
                    # TODO: stopping criterion!!!
                    self.parameters[split] = global_weights
                    self.workflows_states[split] = self.iteration >= self.federated_hyper_parameters["max_iter"]
            super(DeepApp, self).global_aggregation()

    def write_results(self):
        self.progress = "write"
        for split in self.splits.keys():
            path = split.replace("input", "output")
            y_pred, y_true = self.client_model.predict(self.data_loaders[split].get_test_loader())
            pd.DataFrame(y_pred, columns=['y_pred']).to_csv(f"{path}/{self.pred_filename}", index=None)
            pd.DataFrame(y_true, columns=['y_true']).to_csv(f"{path}/{self.y_true_filename}", index=None)
        super(DeepApp, self).write_results()


logic = DeepApp()

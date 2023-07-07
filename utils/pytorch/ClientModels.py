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


class ClientModels:
    """client model with a specific DNN and dataset
    attributes.
    ----------
    train_all: bool
    model:DeepModels
    num_batches_to_train: int
    data_loader:
    generator:
    num_samples : int
        number of samples in the training set
    num_trained_samples: int
        number of unique samples used in the last update call

    Methods
    -------
    update(weights, verbose)
        train the DNN model.
    single_batch_update(weights, verbose)
        update the model for limited batches
    assign_data(x, y, mean, std)
        Normalize and store the dataset inside the model.
    get_weights()
    set_weights(self, weights)
    evaluate(data_loader)
    predict(data_loader)

    """

    def __init__(self, model, aggregator=None):
        """
        Parameters
        ----------
        model:
        train_all:
        batch_count:
        """
        self.model = model
        self.aggregator = aggregator
        self.num_trained_samples = 0

    def update(self, data_loader, global_updates, backup, test_loader, tsboard):
        """update client's model for all of its data for E epochs
            and decreases the learning rate of DNN.

        Parameters
        ----------
        data_loader: DataLoader
            Custom data loader
        global_updates : list
            weights of Deep Neural Network
        backup: dict
            Pytorch Optimizers Parameters
        verbose: bool
            printing an evaluation report
        """
        # global_updates = self.interpret_global_updates(global_updates)
        self.model.set_weights(global_updates['weights'])
        self.model.set_optimizer_params(backup)

        self.model.fit(data_loader, test_loader, global_updates['config'], tsboard, **{})

    def get_local_updates(self, l_update_schema):
        """ Get the local parameters, n_samples, etc. that are needed by the aggregator

        """

        weights, gradients, n_samples, cross_validation = l_update_schema
        local_updates = []
        if weights:
            local_updates.append(self.model.get_weights())
        if gradients:
            local_updates.append(self.model.get_gradients())
        if n_samples:
            local_updates.append(self.model.n_trained_samples)
        return local_updates

    def local_backup(self):
        return self.get_optimizer_params()

    def get_weights(self):
        """ get model's weights

        Returns
        -------
        weights: list

        """
        return self.model.get_weights()

    def set_weights(self, weights):
        """ set model's weights

        Parameters
        ----------
        weights: list

        """
        self.model.set_weights(weights)

    def evaluate(self, test_loader):
        """ Calls model's evaluate method

        Parameters
        ----------
        test_loader: DataLoader
            Custom data loader

        Returns
        -------
        loss_and_accuracy: list

        """
        self.model.evaluate(test_loader)
        self.model.metrics.logs(train=False)
        return self.model.metrics.dict()

    def predict(self, test_loader=None):
        """ Calls model's predict method

        Parameters
        ----------


        Returns
        -------
        predicted_labels: list
        True_labels: list

        """
        if hasattr(self, "test_loader"):
            return self.model.predict(self.test_loader)
        elif test_loader is not None:
            return self.model.predict(test_loader)
        return [], []

    def get_optimizer_params(self):
        return self.model.get_optimizer_params()

    def set_optimizer_params(self, state_dict):
        self.model.set_optimizer_params(state_dict)

    def store_model(self, path):
        self.model.store_model(path)

    def load_model(self, path):
        self.model.load_model(path)

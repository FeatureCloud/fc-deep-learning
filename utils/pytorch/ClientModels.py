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

    def __init__(self, model, train_all, batch_count):
        """
        Parameters
        ----------
        model:
        train_all:
        batch_count:
        """
        self.model = model
        self.train_all = train_all
        self.batch_count = batch_count
        self.num_trained_samples = 0

    def update(self, data_loader, weights, state_dict, verbose=False):
        """update client's model for all of its data for E epochs
            and decreases the learning rate of DNN.

        Parameters
        ----------
        data_loader: DataLoader
            Custom data loader
        weights : list
            weights of Deep Neural Network
        state_dict: dict
            Pytorch Optimizers Parameters
        verbose: bool
            printing an evaluation report
        """
        self.model.set_weights(weights)
        self.model.set_optimizer_params(state_dict)
        if self.train_all:
            self.model.fit(data_loader, verbose=verbose)
            self.num_trained_samples = len(data_loader)
        else:
            _, _, self.num_trained_samples = self.model.train_on_batches(data_loader, self.batch_count, verbose)

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
        return self.model.evaluate(test_loader)

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

    def store(self, path):
        self.model.store(path)
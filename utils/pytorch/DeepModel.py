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
import torch
import numpy as np
import abc
from utils.pytorch.utils import LocalUpdates, Metrics
from utils.utils import to_numpy


class Trainer(abc.ABC):
    """ Deep Convolutional Network
    Attributes
    ----------
    loss_func : torch.nn.modules.loss.CrossEntropyLoss
    device : str
        device name to pass the model parameters to.
    model: CNN.Model
    loss_func: nn.CrossEntropyLoss()
    optimizer:


    Methods
    -------
    evaluate(x, y)
        evaluate the network's performances
    train_on_batch(x, y)
        train the network on entire data in one pass
    get_weights()
        invokes the get_wights method of CNN or MLP class
    set_weights(w)
        invokes the set_weights method of CNN or MLP class
    predict(dl)
        Predict labels of samples in the data loader

    """

    def __init__(self, model, config, attributes, train_config, req_local_updates, log):
        """

        Parameters
        ----------
        model: CNN.Model
        config: dict
            arguments for the model
        attributes: dict
            arguments for DeepModel
        """
        for k, v in train_config.items():
            setattr(self, k, v)

        self.log = log
        metrics = attributes.pop('metrics')
        self.metrics = Metrics(metrics, self.device)
        self.train_metrics_hist, self.test_metrics_hist = [], []


        self.model = model(**config)
        self.model.to(device=self.device)

        # initialize loss
        self.loss_func = self.loss_instance(attributes)

        # initialize optimizer
        self.optimizer = self.opt_instance(attributes)

        self.req_local_updates = req_local_updates
        self.n_trained_samples = None

    def opt_instance(self, attributes):
        opt = attributes.pop('optimizer')
        opt_func = opt['opt']
        opt_param = opt.get('param', {})
        opt_param.update({'params': self.model.parameters()})
        return opt_func(**opt_param)

    def loss_instance(self, attributes):
        loss = attributes.pop('loss')
        loss_func = loss['func']
        loss_param = loss.get('param', {})
        return loss_func(**loss_param).to(device=self.device)

    def get_module(self, module_class, module, params, to_device=False):
        mod = getattr(module_class, module)(**params)
        if to_device:
            mod = mod.to(device=self.device)
        return mod

    def evaluate(self, dl):
        """ evaluate the network's performances in terms of loss and accuracy
            load input numpy arrays in a DataLoader and split it into batches

        Parameters
        ----------
        dl :

        Returns
        -------
        loss : float
            running loss value
        acc : float
            running accuracy
        """

        self.metrics.reset()
        self.model.eval()
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                pred = self.model(data)
                loss = self.loss_func(pred, target)
                self.metrics.perform(pred, target, loss.item())

        self.model.train()

    def predict(self, dl):
        """ evaluate the network's performances in terms of loss and accuracy
            load input numpy arrays in a DataLoader and split it into batches

        Parameters
        ----------
        dl :

        Returns
        -------
        loss : float
            running loss value
        acc : float
            running accuracy
        """
        self.model.eval()
        prediction = []
        y_true = []
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                pred = self.model(data).max(1)[1]
                for item in pred:
                    prediction.append(item.item())
                for item in target:
                    y_true.append(item.item())
        self.model.train()
        return np.array(prediction, dtype='int'), np.array(y_true, dtype='int')

    @abc.abstractmethod
    def train_on_batch(self, data, targets):
        """ train the network on entire data in one pass

        Parameters
        ----------
        data : numpy.ndarray
            image samples
        targets : numpy.array
            labels of the image samples

        """

    @abc.abstractmethod
    def fit(self, train_loader, validation=None, train_config=None, **kwargs):
        """

        Parameters
        ----------
        train_config
        train_loader
        validation
        verbose

        Returns
        -------

        """

    def per_epoch_validation(self, validation, epoch):
        if validation is not None:
            self.evaluate(validation)
            if self.verbose:
                logs = self.metrics.logs(epoch=epoch, train=False)
                self.log(logs)
        elif self.verbose:
            self.log(self.metrics.logs(epoch=epoch))

    def get_weights(self):
        """ Call get_weights method of CNN or MLP classes

        """
        with torch.no_grad():
            w = []
            for name, param in self.model.named_parameters():
                w.append(param.data.clone().detach().cpu().numpy())
        return w

    def set_weights(self, w):
        """ Call set_weights method of CNN or MLP classes

        Parameters
        ----------
        w : numpy.array
            networks weights with arbitrary dimensions

        """
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                # p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
                # p = to_numpy(w[i], skip_obj_dtype=True)
                p = to_numpy(w[i], skip_obj_dtype=True)
                # print(p.dtype)
                param.data = torch.FloatTensor(p).to(device=self.device)

    def get_gradients(self):
        # TODO
        pass

    def set_global_updates(self, updates):
        """ Call set_weights method of CNN or MLP classes

        Parameters
        ----------
        updates : numpy.array
            networks weights with arbitrary dimensions

        """

    def local_backup(self):
        backup = [
            self.model.getoptimizer_params(),
        ]
        return backup

    def set_optimizer_params(self, param):
        self.optimizer.load_state_dict(param)

    def get_optimizer_params(self):
        return self.optimizer.state_dict()

    def store_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class BasicTrainer(Trainer):
    def __init__(self, **kwargs):
        super(BasicTrainer, self).__init__(**kwargs)
        self.n_trained_samples = 0

    def train_on_batch(self, data, targets, **kwargs):
        """ train the network on entire data in one pass

        Parameters
        ----------
        data : numpy.ndarray
            image samples
        targets : numpy.array
            labels of the image samples

        """
        data, targets = data.to(self.device), targets.to(self.device)
        pred = self.model(data)
        loss = self.loss_func(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.metrics.perform(pred, targets, loss.item())

    def apply_grads(self, grads):
        self.optimizer.zero_grad()
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g
        self.optimizer.step()

    def fit(self, train_loader, validation=None, train_config=None):
        """

        Parameters
        ----------
        train_config
        train_loader
        validation
        verbose

        Returns
        -------

        """
        for e in range(self.epochs):
            self.metrics.reset()
            for i, data in enumerate(train_loader):
                self.train_on_batch(data[0], data[1])
            self.log(self.metrics.logs(epoch=e))
            self.per_epoch_validation(validation, e + 1)
        self.n_trained_samples = len(train_loader)

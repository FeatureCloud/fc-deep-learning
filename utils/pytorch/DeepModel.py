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
import torch.optim as optim
import torch
import numpy as np


class Model:
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

    def __init__(self, model, config, attributes, device):
        """

        Parameters
        ----------
        model: CNN.Model
        config: dict
            arguments for the model
        attributes: dict
            arguments for DeepModel
        """

        self.metrics = {}
        self.metrics_initialize(attributes.pop('metrics'))

        # Set attributes for Model:
        # batch_size
        # epochs
        for k, v in attributes.items():
            setattr(self, k, v)
        self.device = device
        self.model = model(**config)
        self.model.to(device=self.device)

        # initialize loss
        loss = attributes.pop('loss')
        loss_func = loss['func']
        loss_param = loss.get('param', {})
        self.loss_func = loss_func(**loss_param).to(device=self.device)

        # initialize optimizer
        opt = attributes.pop('optimizer')
        opt_func = opt['opt']
        opt_param = opt.get('param', {})
        opt_param.update({'params': self.model.parameters()})
        self.optimizer = opt_func(**opt_param)

    def metrics_initialize(self, metric_classes):
        for metric in metric_classes:
            m = {'func': metric['func'](**metric.get('param', {})),
                 'AverageMeter': AverageMeter()
                 }
            self.metrics[metric['name']] = m

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
        test_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                pred = self.model(data)
                loss = self.loss_func(pred, target)
                self.metric_performance(pred, target)
                test_loss.update(loss.item(), data.size(0))
        self.model.train()
        return test_loss.avg

    def metric_performance(self, pred, target):
        for name, metric in self.metrics.items():
            perf = metric['func'](pred, target)
            metric['AverageMeter'].update(perf)

    def metric_reset(self):
        for metric in self.metrics.values():
            metric['AverageMeter'].reset()

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

    def train_on_batch(self, data, targets):
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
        self.metric_performance(pred, targets)
        train_loss = loss.detach().item()
        return train_loss

    def fit(self, train_loader, validation=None, verbose=False):
        """

        Parameters
        ----------
        train_loader
        validation
        verbose

        Returns
        -------

        """
        train_loss = AverageMeter()
        total_loss_train, total_loss_test = [], []
        for e in range(self.epochs):
            self.metric_reset()
            train_loss.reset()
            for i, data in enumerate(train_loader):
                loss = self.train_on_batch(data[0], data[1])
                train_loss.update(loss, data[0].size(0))
            if validation is not None:
                loss = self.evaluate(validation)
                total_loss_test.append(loss)
                if verbose:
                    train_logs = self.metrics_logs(train_loss.avg, epoch=e + 1)
                    test_logs = self.metrics_logs(loss, train=False)
                    print(f"{train_logs} {test_logs}")
            elif verbose:
                print(self.metrics_logs(train_loss.avg, epoch=e + 1))
            total_loss_train.append(train_loss.avg)
        return total_loss_train, total_loss_test

    def metrics_logs(self, loss, epoch=None, train=True):
        phase = "train" if train else "test"
        msg = ""
        if epoch is not None:
            msg = f"Epoch {epoch}: "
        msg = f"{msg}{phase}_loss: {loss:.4f}"
        for name, metric in self.metrics.items():
            msg = f"{msg} {phase}_{name}: {metric['AverageMeter'].avg:.4f}"
        return msg

    def train_on_batches(self, train_loader, n_batches, verbose=False):
        """

        Parameters
        ----------
        train_loader:  DataLoader
            Custom data loader
        n_batches: int
        verbose: bool

        Returns
        -------

        """
        train_loss, train_acc = AverageMeter(), AverageMeter()
        n_trained_samples = 0
        for i, (data, target) in enumerate(train_loader):
            loss, acc = self.train_on_batch(data, target)
            train_acc.update(acc, data[0].size(0))
            train_loss.update(loss, data[0].size(0))
            n_trained_samples += len(data)
            if i == n_batches - 1:
                break
        if verbose:
            print(f"Loss= {train_loss.avg:.2f}, Accuracy={train_acc.avg:.2f}")
        return train_loss.avg, train_acc.avg, n_trained_samples

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
                p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
                param.data = torch.from_numpy(p).to(device=self.device)

    def set_optimizer_params(self, param):
        self.optimizer.load_state_dict(param)

    def get_optimizer_params(self):
        return self.optimizer.state_dict()

    def store_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

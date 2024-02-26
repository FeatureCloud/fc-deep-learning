from utils.pytorch.utils import GlobalUpdates
import abc
import numpy as np
from utils.utils import to_list, to_numpy


class FedOptimizer(abc.ABC):
    """
        weights should be ready after the aggregation
        other updates should be ready after post_aggregation

    """

    def __init__(self, global_updates, train_config, max_iter, **kwargs):
        self.global_updates = getattr(GlobalUpdates, global_updates).value
        self.train_config = train_config
        self.max_iter = max_iter
        self.iteration = 0

    @abc.abstractmethod
    def aggregate(self, params, **kwargs):
        """ Federated aggregation

        Parameters
        ----------
        params
        kwargs

        Returns
        -------

        """

    @abc.abstractmethod
    def aggregate_smpc(self, params):
        """ Aggregate results of SMPC

        Parameters
        ----------
        params

        Returns
        -------

        """

    @abc.abstractmethod
    def post_aggregate(self):
        """

        Returns
        -------

        """

    @property
    @abc.abstractmethod
    def stoppage(self):
        """

        Returns
        -------

        """

    @property
    @abc.abstractmethod
    def weights(self):
        """

        Returns
        -------

        """

    @property
    @abc.abstractmethod
    def gradients(self):
        """

        Returns
        -------

        """

    @property
    @abc.abstractmethod
    def config(self):
        """

        Returns
        -------

        """

    def get_global_updates(self):
        """ Share global updates based on

        Returns
        -------

        """
        weights, gradients, config, stopping, cross_validation = self.global_updates
        local_updates = []
        if weights:
            # self.weights = self.get_weights()
            local_updates.append(to_list(self.weights))
        if gradients:
            # self.gradients = self.get_gradients()
            local_updates.append(self.gradients)
        if config:
            local_updates.append(self.config)
        if stopping:
            local_updates.append(self.stoppage)
        return local_updates


class FedAvg(FedOptimizer):
    """
    Supporting:
    * SMPC
    * CrossValidation
    * ReqAggData.WEIGHTS_N_SAMPLES
    """

    def __init__(self, **kwargs):
        super(FedAvg, self).__init__(**kwargs)
        self.global_weights = []
        self.stopping_criteria = []

    def aggregate(self, params, **kwargs):
        self.iteration += 1
        n_splits = len(params[0])
        global_weights = [np.array(params[0][0][0], dtype='object') * 0] * n_splits
        total_n_samples = [0] * n_splits
        for client_models in params:
            for model_counter, (weights, n_samples) in enumerate(client_models):
                global_weights[model_counter] += np.array(weights, dtype='object') * n_samples
                total_n_samples[model_counter] += n_samples
        updated_weights = []
        for counter, (w, n) in enumerate(zip(global_weights, total_n_samples)):
            updated_weights.append(w / n)
        self.global_weights = updated_weights

    def aggregate_smpc(self, params):
        global_weights = [to_numpy(model) / total_n_sample for model, total_n_sample in params]
        self.weights = global_weights

    def post_aggregate(self, **kwargs):
        metrics = kwargs['metrics']
        iter_limit = self.iteration >= self.max_iter
        self.stopping_criteria = [[iter_limit]] * len(metrics)

    @property
    def stoppage(self):
        return self.stopping_criteria

    @property
    def weights(self):
        return self.global_weights

    @property
    def config(self):
        return None

    @property
    def gradients(self):
        return None

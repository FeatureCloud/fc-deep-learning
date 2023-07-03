from utils.pytorch.optimizer import FedOptimizer
import numpy as np
from utils.utils import to_numpy


class CustomAggregator(FedOptimizer):
    """
    Supporting:
    * SMPC
    * CrossValidation
    * ReqAggData.WEIGHTS_N_SAMPLES
    """
    def __init__(self, **kwargs):
        super(CustomAggregator, self).__init__(**kwargs)
        self.global_weights = []
        self.stopping_criteria = []

    def aggregate(self, params, **kwargs):
        n_splits = len(params[0])
        global_weights = [np.array(params[0][0][0], dtype='object') * 0] * n_splits
        for client_models in params:
            for model_counter, weights in enumerate(client_models):
                global_weights[model_counter] += np.array(weights, dtype='object')

        updated_weights = []
        for counter, w in enumerate(global_weights):
            updated_weights.append(w / len(global_weights))
        self.global_weights = updated_weights

    def aggregate_smpc(self, params, n_clients):
        global_weights = [to_numpy(model) / n_clients for model in params]
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

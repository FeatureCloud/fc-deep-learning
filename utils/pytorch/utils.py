from enum import Enum
import pandas as pd


class LocalUpdates(Enum):
    """ The required data items for the aggregation model
        After local update, the client model will communicate the update parts accordingly.

        Order: (WEIGHTS: bool, GRADIENTS: bool, N_SAMPLES: bool, CROSS_VALIDATION: bool)

    """
    WEIGHTS = (True, False, False, False)
    GRADIENTS = (False, True, False, False)
    WEIGHTS_N_SAMPLES = (True, False, True, False)
    GRADIENTS_N_SAMPLES = (False, True, True, False)
    WEIGHTS_CROSS_VALIDATION = (True, False, False, True)
    GRADIENTS_CROSS_VALIDATION = (False, True, False, True)
    WEIGHTS_N_SAMPLES_CROSS_VALIDATION = (True, False, True, True)
    GRADIENTS_N_SAMPLES_CROSS_VALIDATION = (False, True, True, True)
    ALL = (True, True, True, True)


class GlobalUpdates(Enum):
    """ The required global data items that corrdinator will broadcat.
        After Global aggregation, the client model will communicate the update parts accordingly.

    Order (WEIGHTS, GRADIENTS, CONFIG, STOPPING, CROSS_VALIDATION)
    """
    WEIGHTS = (True, False, False, False, False)
    GRADIENTS = (False, True, False, False, False)
    WEIGHTS_CONFIG = (True, False, True, False, False)
    GRADIENTS_CONFIG = (False, True, True, False, False)
    WEIGHTS_CROSS_VALIDATION = (True, False, False, False, True)
    GRADIENTS_CROSS_VALIDATION = (False, True, False, False, True)
    WEIGHTS_CONFIG_CROSS_VALIDATION = (True, False, True, False, True)
    GRADIENTS_CONFIG_CROSS_VALIDATION = (False, True, True, False, True)
    WEIGHTS_STOPPING = (True, False, False, True, False)
    GRADIENTS_STOPPING = (False, True, False, True, False)
    WEIGHTS_CONFIG_STOPPING = (True, False, True, True, False)
    GRADIENTS_CONFIG_STOPPING = (False, True, True, True, False)
    WEIGHTS_STOPPING_CROSS_VALIDATION = (True, False, False, True, True)
    GRADIENTS_STOPPING_CROSS_VALIDATION = (False, True, False, True, True)
    WEIGHTS_CONFIG_STOPPING_CROSS_VALIDATION = (True, False, True, True, True)
    GRADIENTS_CONFIG_STOPPING_CROSS_VALIDATION = (False, True, True, True, True)


class Metrics:
    def __init__(self, metric_classes):
        self.metric_classes = metric_classes
        self.metrics = self.init_metrics()
        self.loss = AverageMeter()
        self.df = pd.DataFrame(data={name: [] for name in self.metrics.keys()})
        self.records = []

    def init_metrics(self):
        metrics = {}
        for metric in self.metric_classes:
            m = {'func': metric['func'](**metric.get('param', {})),
                 'AverageMeter': AverageMeter()
                 }
            metrics[metric['name']] = m
        return metrics

    def reset(self):
        self.records.append(self.dict())
        for metric in self.metrics.values():
            metric['AverageMeter'].reset()
        self.loss.reset()

    def perform(self, pred, target, loss):
        d_size = len(target)
        self.loss.update(loss, d_size)
        for name, metric in self.metrics.items():
            perf = metric['func'](pred, target)
            metric['AverageMeter'].update(perf, d_size)

    def logs(self, epoch=None, train=True):
        phase = "train" if train else "test"
        msg = ""
        if epoch is not None and train:
            msg = f"Epoch {epoch}: "
        msg = f"{msg}{phase}_loss: {self.loss.avg:.4f}"
        for name, metric in self.metrics.items():
            msg = f"{msg} {phase}_{name}: {metric['AverageMeter'].avg:.4f}"
        return msg

    def tabular(self):
        return pd.DataFrame(data={name: [metric['AverageMeter'].avg] for name, metric in self.metrics.items()})

    def dict(self):
        return {name: metric['AverageMeter'].avg for name, metric in self.metrics.items()}

    def to_df(self):
        return pd.DataFrame.from_records(self.records)



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

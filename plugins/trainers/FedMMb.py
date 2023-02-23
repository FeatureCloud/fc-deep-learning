from utils.pytorch.DeepModel import BasicTrainer


class CustomTrainer(BasicTrainer):
    def __init__(self, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.n_trained_samples = 0

    def fit(self, train_loader, validation=None, train_config=None, **kwargs):
        """

        Parameters
        ----------
        train_config
        train_loader
        validation

        Returns
        -------

        """
        n_trained_samples = 0
        for i, (data, target) in enumerate(train_loader):
            self.train_on_batch(data, target)
            n_trained_samples += len(data)
            if i == self.batch_count - 1:
                break
        self.log(self.metrics.logs(epoch=0))
        self.per_epoch_validation(validation, epoch=1)
        return n_trained_samples
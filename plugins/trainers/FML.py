# Federated Mutual Learning
from copy import deepcopy
from utils.pytorch.DeepModel import Trainer
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax
import torch


class CustomTrainer(Trainer):
    def __init__(self, meme_model, meme_config, meme_attributes, model, config, attributes, train_config, req_local_updates, log):
        """ Training two models in the DML fashion: meme and local
            - meme model is trained as the self.model:
                * model: self.model
                * optimizer: self.optimizer
                * metrics: self.metrics

            - local model is trained as the self.local_model:
                * model: self.local_model
                * optimizer: self.local_optimizer
                * metrics: self.local_metrics


        Parameters
        ----------
        meme_model
        meme_config
        kwargs: Should include `alpha` & `beta` parameters for the FML loss
        """
        super().__init__(meme_model, meme_config, meme_attributes, train_config, req_local_updates, log)
        self.local_metrics = deepcopy(self.metricts)
        self.local_model = meme_model(**config).to(device=self.device)

        # initialize optimizer for meme
        self.local_optimizer = self.opt_instance(meme_attributes)
        # Both models use the same loss
        self.kl_loss = KLDivLoss(reduction='batchmean')
        # self.teacher_req_local_updates = teacher_req_local_updates

    # def copy_model_to_student(self):
    #     self.student_model = deepcopy(self.model)
    #
    # def copy_model_to_teacher(self):
    #     self.teacher_model = deepcopy(self.model)
    #
    # def copy_student_to_model(self):
    #     self.model = deepcopy(self.student_model)
    #
    # def copy_teacher_to_model(self):
    #     self.model = deepcopy(self.teacher_model)

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
        meme_logits = self.model(data)
        local_logits = self.local_model(data)
        meme_loss, local_loss = self.fml_loss(local_logits, meme_logits, targets)

        # updating the local model



        # updating the meme model

        local_loss.backward()
        self.local_optimizer.step()
        self.local_optimizer.zero_grad()
        self.local_metrics.perform(local_logits, targets, local_loss.item())

        meme_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.metrics.perform(meme_logits, targets, meme_loss.item())

        # TODO: extend metrics

    def fml_loss(self, local_logits, meme_logits, targets):
        local_loss = self.alpha * self.loss_func(local_logits, targets) + \
                     (1 - self.alpha) * self.kl_loss(log_softmax(meme_logits, dim=1), log_softmax(local_logits, dim=1))

        meme_loss = self.beta * self.loss_func(meme_logits, targets) + \
                    (1 - self.beta) * self.kl_loss(log_softmax(local_logits, dim=1), log_softmax(meme_logits, dim=1))
        return local_loss, meme_loss

    def fit(self, train_loader, validation=None, train_config=None, **kwargs):
        self.local_model = kwargs['teachers_models']
        for e in range(self.epochs):
            self.metrics.reset()
            self.local_metrics.reset()
            for i, data in enumerate(train_loader):
                self.train_on_batch(data[0], data[1])
            self.log("\t********\tmeme models' performance\t********")
            self.log(self.metrics.logs(epoch=e))
            self.log("\t********\tLocal models' performance\t********")
            self.log(self.local_metrics.logs(epoch=e))

            self.per_epoch_validation(validation, e + 1)
        self.n_trained_samples = len(train_loader)

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
        super(CustomTrainer, self).evaluate(dl)
        self.local_metrics.reset()
        self.local_model.eval()
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                target = target.to(device=self.device)
                pred = self.local_model(data)
                loss = self.loss_func(pred, target)
                self.local_metrics.perform(pred, target, loss.item())

        self.local_model.train()

    def per_epoch_validation(self, validation, epoch):
        if validation is not None:
            self.evaluate(validation)
            if self.verbose:
                self.log("\t********\tLocal models' performance on testset\t********")
                logs = self.local_metrics.logs(epoch=epoch, train=False)
                self.log(logs)

                self.log("\t********\tmeme models' performance on testset\t********")
                logs = self.metrics.logs(epoch=epoch, train=False)
                self.log(logs)

        elif self.verbose:
            self.log("\t********\tLocal models' performance\t********")
            self.log(self.local_metrics.logs(epoch=epoch))

            self.log("\t********\tmeme models' performance\t********")
            self.log(self.metrics.logs(epoch=epoch))

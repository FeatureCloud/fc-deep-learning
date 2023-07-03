from utils.pytorch.ClientModels import ClientModels


class CostumeClient(ClientModels):
    def __init__(self, teacher_model, **kwargs):
        super().__init__(kwargs)
        # self.model serves as student model or private model
        self.teacher_model = teacher_model

    def update(self, data_loader, global_updates, backup, test_loader):
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

        self.model.fit(data_loader, test_loader, global_updates['config'], **{})

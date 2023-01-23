import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None):
        super(CustomLoss, self).__init__()
        self.gamma = gamma
        self.weight = torch.FloatTensor(weight)

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

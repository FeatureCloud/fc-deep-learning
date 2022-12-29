import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, n_classes)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.act(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        return x

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
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_channels, img_width, n_classes):
        """

        Parameters
        ----------
        n_channels : int
            number of channels
        img_width : int
            width (or height) of the sample images
        n_classes : int
            number of class labels
        """
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(img_width // 4 * img_width // 4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, n_classes)

    def forward(self, x):
        """ feed forward x through the network

        Parameters
        ----------
        x : torch.Tensor
            data to feedforward

        Returns
        -------
        x : torch.Tensor
            the output of the network
        """
        x = F.relu(self.conv1(x))
        x = self.pool2x2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2x2(x)
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def num_flat_features(x):
    """ calculating the size of x for all dimensions
        except the first one (batch).

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    num_features : int
        number of elements of x(except first dim.)
    """
    size = x.size()[1:]  #
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

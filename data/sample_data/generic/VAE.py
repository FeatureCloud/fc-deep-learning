import torch
import torch.nn.functional as F

from collections import OrderedDict


class MeanAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), 1e-5, 1e6)


class DispAct(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        return torch.clamp(self.softplus(x), 1e-4, 1e4)


def init_weights(layer):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        # torch.nn.init.zeros_(layer.bias)


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__()

        self.input_size = input_size
        self.encoder_size = encoder_size
        self.bottleneck_size = bottleneck_size

        self.encoder = torch.nn.Sequential(
            OrderedDict([
                ('linear1', torch.nn.Linear(self.input_size, self.encoder_size, bias=True)),
                # ('bn1', torch.nn.BatchNorm1d(self.encoder_size)),
                ('ln1', torch.nn.LayerNorm(self.encoder_size)),
                # ('gn1', torch.nn.GroupNorm(4, self.encoder_size)),
                ('relu1', torch.nn.ReLU())
            ]))

        self.bottleneck = torch.nn.Sequential(
            OrderedDict([
                ('linear2', torch.nn.Linear(self.encoder_size, self.bottleneck_size, bias=True)),
                # ('bn2', torch.nn.BatchNorm1d(self.bottleneck_size)),
                ('ln2', torch.nn.LayerNorm(self.bottleneck_size)),
                # ('gn2', torch.nn.GroupNorm(4, self.bottleneck_size)),
                ('relu2', torch.nn.ReLU())
            ]))

        self.decoder = torch.nn.Sequential(
            OrderedDict([
                ('linear3', torch.nn.Linear(self.bottleneck_size, self.encoder_size, bias=True)),
                # ('bn3', torch.nn.BatchNorm1d(self.encoder_size)),
                ('ln3', torch.nn.LayerNorm(self.encoder_size)),
                # ('gn3', torch.nn.GroupNorm(4, self.encoder_size)),
                ('relu3', torch.nn.ReLU())
            ]))

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x


class NBAutoEncoder(AutoEncoder):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__(input_size, encoder_size, bottleneck_size)

        self.mean = torch.nn.Sequential(
            OrderedDict([
                ('linear_m', torch.nn.Linear(self.encoder_size, self.input_size)),
                ('meanact', MeanAct())
            ]))

        self.disp = torch.nn.Sequential(
            OrderedDict([
                ('linear_di', torch.nn.Linear(self.encoder_size, self.input_size)),
                ('dispact', DispAct())
            ]))

        self.mean.apply(init_weights)
        self.disp.apply(init_weights)

    def forward(self, x, sf):
        x = super().forward(x)

        mean_norm = self.mean(x)
        disp = self.disp(x)

        mean = mean_norm * torch.reshape(sf, (sf.shape[0], 1))

        return mean, disp


class ZINBAutoEncoder(AutoEncoder):
    def __init__(self, input_size, encoder_size, bottleneck_size):
        super().__init__(input_size, encoder_size, bottleneck_size)

        self.mean = torch.nn.Sequential(
            OrderedDict([
                ('linear_m', torch.nn.Linear(self.encoder_size, self.input_size)),
                ('meanact', MeanAct())
            ]))

        self.disp = torch.nn.Sequential(
            OrderedDict([
                ('linear_di', torch.nn.Linear(self.encoder_size, self.input_size)),
                ('dispact', DispAct())
            ]))

        self.drop = torch.nn.Sequential(
            OrderedDict([
                ('linear_dr', torch.nn.Linear(self.encoder_size, self.input_size)),
                ('sigmoid', torch.nn.Sigmoid())
            ]))

        self.mean.apply(init_weights)
        self.disp.apply(init_weights)
        self.drop.apply(init_weights)

    def forward(self, x, sf):
        x = super().forward(x)

        mean_norm = self.mean(x)
        disp = self.disp(x)
        drop = self.drop(x)

        mean = mean_norm * torch.reshape(sf, (sf.shape[0], 1))

        return mean, disp, drop


class Classifier(torch.nn.Module):
    def __init__(self, input_size, conv_out_size, kernel_size, num_classes):
        super().__init__()

        self.conv = torch.nn.Conv1d(input_size, conv_out_size, kernel_size)
        self.pool = torch.nn.MaxPool1d(kernel_size)
        self.dense = torch.nn.Linear(conv_out_size, num_classes)
        self.sm = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        data = self.relu(self.conv(data.unsqueeze(2)))
        data = self.pool(data)
        data = self.dense(data.squeeze(2))
        data = self.sm(data)

        return data
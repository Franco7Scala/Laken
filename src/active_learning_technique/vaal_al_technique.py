import numpy
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
import src.support.support as support

from torch.utils.data import DataLoader
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique
from src.al_dataset.dataset import Dataset
from copy import deepcopy


class VAALALTechnique(AbstractALTechnique):

    def __init__(self, neural_network, vae, dataset, n_classes):
        self.neural_network = neural_network
        self.vae = vae
        self.discriminator = Discriminator(vae.device, vae.dim_code)
        self.dataset = dataset
        self.n_classes = n_classes

    def select_samples(self, unlabeled_samples, n_samples_to_select):
        all_preds = []
        all_indices = []

        for sample in unlabeled_samples:
            with torch.no_grad():
                mu, _, _ = self.vae(torch.unsqueeze(sample.to(self.vae.device), 0))
                pred = self.discriminator(mu)

            pred = pred.cpu().data
            all_preds.extend(pred)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds *= -1
        _, query_indices = torch.topk(all_preds, n_samples_to_select)
        return [unlabeled_samples[i] for i in query_indices]


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Discriminator(nn.Module):

    def __init__(self, device, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()
        self.device = device
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

import numpy
import torch
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
        self.discriminator = Discriminator(vae.dim_code)
        self.dataset = dataset
        self.n_classes = n_classes

    def select_samples(self, unlabeled_samples, n_samples_to_select):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            with torch.no_grad():
                mu, _, _ = self.vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]
        return querry_pool_indices


class Discriminator(nn.Module):

    def __init__(self, z_dim=10):
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

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)

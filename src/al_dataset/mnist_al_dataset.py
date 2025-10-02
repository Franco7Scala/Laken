import src.support.support as support

from src.al_dataset.abstract_al_dataset import AbstractALDataset
from torchvision import datasets, transforms


class MNISTALDataset(AbstractALDataset):

    def __init__(self, percentage_labeled):
        self.test_dataset = datasets.MNIST(root=support.dataset_path.format("mnist"), train=False, transform=transforms.ToTensor(), download=True)
        self.train_dataset = datasets.MNIST(root=support.dataset_path.format("mnist"), train=True, transform=transforms.ToTensor(), download=True)
        super(MNISTALDataset, self).__init__(percentage_labeled, self.test_dataset, self.train_dataset, (28, 28))

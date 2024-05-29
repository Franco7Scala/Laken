import src.support as support

from src.al_dataset.abstract_al_dataset import AbstractALDataset
from torchvision import datasets, transforms


class FashionMNISTALDataset(AbstractALDataset):

    def __init__(self, desired_counts):
        test_dataset = datasets.FashionMNIST(root=support.dataset_path.format("fashion_mnist"), train=False, transform=transforms.ToTensor(), download=True)
        train_dataset = datasets.FashionMNIST(root=support.dataset_path.format("fashion_mnist"), train=True, transform=transforms.ToTensor(), download=True)
        super(FashionMNISTALDataset, self).__init__(test_dataset, train_dataset, (28, 28), desired_counts)

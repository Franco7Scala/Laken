import sys
import torch
import src.support.support as support

from torchvision import datasets, transforms
from src.neural_networks.fashion_mnist.fashion_mnist_vae import Fashion_MNIST_VAE, loss_vae
from src.neural_networks.mnist.mnist_vae import MNIST_VAE
from src.support.support import dataset_path, Reason, clprint


if __name__ == "__main__":
    support.warm_up()

    dataset_name = "mnist"  # "fmnist"


    #############################################################################################################


    clprint("Loading dataset...", Reason.INFO_TRAINING)
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=dataset_path.format("mnist"), train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.MNIST(root=dataset_path.format("mnist"), train=False, transform=transforms.ToTensor(), download=False)

    elif dataset_name == "fmnist":
        train_dataset = datasets.FashionMNIST(root=dataset_path.format("fashion_mnist"), train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.FashionMNIST(root=dataset_path.format("fashion_mnist"), train=False, transform=transforms.ToTensor(), download=False)

    else:
        clprint("Unknown dataset...", Reason.WARNING)
        sys.exit(0)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=support.vae_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=support.vae_batch_size, shuffle=False)

    if dataset_name == "mnist":
        vae = MNIST_VAE(support.vae_dim_code, support.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=support.vae_learning_rate)

    elif dataset_name == "fmnist":
        vae = Fashion_MNIST_VAE(support.vae_dim_code, support.device)
        optimizer = torch.optim.Adam(vae.parameters(), lr=support.vae_learning_rate)

    else:
        clprint("Unknown dataset...", Reason.WARNING)
        sys.exit(0)

    clprint("Training neural network...", Reason.INFO_TRAINING)
    vae.fit(support.vae_training_epochs, loss_vae, optimizer, train_loader, test_loader)
    clprint("Saving neural network...", Reason.INFO_TRAINING)
    vae.save(support.model_path)
    clprint("Drawing results...", Reason.INFO_TRAINING)
    vae.draw_reconstructions(test_dataset, support.reconstruction_image_path)
    vae.draw_latent_space(test_dataset, support.latent_space_distribution_path)
    clprint("Completed!", Reason.OUTPUT_TRAINING)

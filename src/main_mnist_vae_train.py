import torch
import src.support as support

from torchvision import datasets, transforms
from src.neural_networks.mnist.mnist_vae import MNIST_VAE, loss_vae
from src.support import dataset_path, Reason, clprint


if __name__ == "__main__":
    support.warm_up()

    train_dataset = datasets.MNIST(root=dataset_path.format("mnist"), train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root=dataset_path.format("mnist"), train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=support.vae_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=support.vae_batch_size, shuffle=False)

    vae = MNIST_VAE(support.vae_dim_code, support.device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=support.vae_learning_rate)
    clprint("Training neural network...", Reason.INFO_TRAINING)
    vae.fit(support.vae_training_epochs, loss_vae, optimizer, train_loader, test_loader)
    clprint("Saving neural network...", Reason.INFO_TRAINING)
    vae.save(support.model_path)
    clprint("Drawing results...", Reason.INFO_TRAINING)
    vae.draw_reconstructions(test_dataset, support.reconstruction_image_path)
    vae.draw_latent_space(test_dataset, support.latent_space_distribution_path)
    clprint("Completed!", Reason.OUTPUT_TRAINING)

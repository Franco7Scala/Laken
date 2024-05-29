import numpy
import torch
import pandas
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px

from IPython.display import clear_output
from tqdm.autonotebook import tqdm
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class MNIST_VAE(nn.Module):

    def __init__(self, dim_code, device):
        super().__init__()
        self.label = nn.Embedding(10, dim_code)
        self.device = device
        # encoder
        self.encoder = nn.Sequential(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.flatten_mu = nn.Linear(128 * 7 * 7, out_features=dim_code)
        self.flatten_log_sigma = nn.Linear(128 * 7 * 7, out_features=dim_code)
        # decoder
        self.decode_linear = nn.Linear(4, 128 * 7 * 7)
        self.decode_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.decode_1 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.to(self.device)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.flatten_mu(x), self.flatten_log_sigma(x)
        z = self.gaussian_sampler(mu, log_sigma)
        return z

    def gaussian_sampler(self, mu, log_sigma):
        if self.training:
            std = torch.exp(log_sigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decode(self, x):
        x = self.decode_linear(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return reconstruction

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu, log_sigma = self.flatten_mu(x), self.flatten_log_sigma(x)
        z = self.gaussian_sampler(mu, log_sigma)
        x = self.decode_linear(z)
        x = x.view(x.size(0), 128, 7, 7)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return mu, log_sigma, reconstruction

    def _train_epoch(self, criterion, optimizer, data_loader):
        train_losses_per_epoch = []
        self.train()
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(self.device)
            mu, log_sigma, reconstruction = self(x_batch)
            loss = criterion(x_batch.to(self.device).float(), mu, log_sigma, reconstruction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses_per_epoch.append(loss.item())

        return numpy.mean(train_losses_per_epoch), mu, log_sigma, reconstruction

    def evaluate(self, criterion, data_loader):
        val_losses_per_epoch = []
        self.eval()
        with torch.no_grad():
            for x_val, _ in data_loader:
                x_val = x_val.to(self.device)
                mu, log_sigma, reconstruction = self(x_val)
                loss = criterion(x_val.to(self.device).float(), mu, log_sigma, reconstruction)
                val_losses_per_epoch.append(loss.item())

        return numpy.mean(val_losses_per_epoch), mu, log_sigma, reconstruction

    def fit(self, epochs, criterion, optimizer, train_loader, test_loader):
        loss = {"train_loss": [], "val_loss": []}
        with tqdm(desc="Training", total=epochs) as pbar_outer:
            for epoch in range(epochs):
                train_loss, _, _, _ = self._train_epoch(criterion, optimizer, train_loader)
                val_loss, _, _, _ = self.evaluate(criterion, test_loader)
                pbar_outer.update(1)
                loss["train_loss"].append(train_loss)
                loss["val_loss"].append(val_loss)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def draw_reconstructions(self, test_dataset, image_path, size=5):
        clear_output(wait=True)
        plt.figure(figsize=(18, 6))
        for k in range(size):
            ax = plt.subplot(2, size, k + 1)
            img = test_dataset[k][0].unsqueeze(0).to(self.device)
            self.eval()
            with torch.no_grad():
                mu, log_sigma, reconstruction = self(img)

            plt.imshow(img.cpu().squeeze().numpy(), cmap="gray")
            plt.axis("off")
            if k == size // 2:
                ax.set_title("Real")

            ax = plt.subplot(2, size, k + 1 + size)
            plt.imshow(reconstruction.cpu().squeeze().numpy(), cmap="gray")
            plt.axis("off")
            if k == size // 2:
                ax.set_title("Output")

        plt.savefig(image_path)

    def draw_latent_space(self, test_dataset, path_image):
        latent_space = []
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        for x, y in tqdm(test_loader):
            img = x.to(self.device)
            label = y.to(self.device)
            self.eval()
            with torch.no_grad():
                latent = self.encode(img)

            latent = latent.flatten().cpu().numpy()
            sample = {f"Encoded_{i}": encoded for i, encoded in enumerate(latent)}
            sample["label"] = label.item()
            latent_space.append(sample)

        latent_space = pandas.DataFrame(latent_space)
        latent_space["label"] = latent_space["label"].astype(str)
        tsne = TSNE(n_components=2)
        digits_embedded = tsne.fit_transform(latent_space.drop(["label"], axis=1))
        figure = px.scatter(digits_embedded, x=0, y=1, color=latent_space["label"], opacity=0.7, labels={"color": "Digit"}, title="Latent space with t-SNE").for_each_trace(lambda t: t.update(name=t.name.replace("=", ": ")))
        figure.update_traces(marker=dict(size=10, line=dict(width=2,  color="DarkSlateGrey")), selector=dict(mode="markers"))
        figure.update_yaxes(visible=False, showticklabels=False)
        figure.update_xaxes(visible=False, showticklabels=False)
        figure.show()
        figure.write_image(path_image)


def load_model(dim_code, path, device):
    model = MNIST_VAE(dim_code, device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def kl_divergence(mu, log_sigma):
    loss = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return loss


def log_likelihood(x, reconstruction):
    loss = nn.BCELoss(reduction="sum")
    return loss(reconstruction, x)


def loss_vae(x, mu, log_sigma, reconstruction):
    return kl_divergence(mu, log_sigma) + log_likelihood(x, reconstruction)

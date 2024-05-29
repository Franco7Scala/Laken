import numpy
import os
import umap.plot
import torch
import pandas
import auto_encoder
import seaborn
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from cifar10_dataset import Cifar10Dataset
from main_data_distribution_visualization import plot_umap_path, plot_tsne_path, model_path, get_configuration, dataset_path


def load_data(device, limit=-1):
    configuration = get_configuration()
    model = auto_encoder.load_model(model_path, configuration, device)
    model = model.to(device)

    x = []
    y = []

    dataset = Cifar10Dataset(1, dataset_path, configuration.shuffle_dataset)
    data = dataset.training_loader
    dataiter = iter(data)

    if limit != -1:
        i = limit
        samples_quantiy = limit
    else:
        i = len(data)
        samples_quantiy = len(data)

    for batch in dataiter:
        x.append(model.compress(batch[0].to(device)).cpu().squeeze().detach().numpy())
        y.append(batch[1].squeeze().detach().numpy())

        i -= 1
        if i <= 0:
            break

    sample_size = (numpy.array(x).shape[1] * numpy.array(x).shape[2])
    
    #return numpy.array(x).reshape((numpy.array(x).shape[0], sample_size)), numpy.array(y)
    return numpy.array(x).reshape(samples_quantiy, sample_size), numpy.array(y)


def plot(device, appendix_name = "none"):
    x, y = load_data(device)

    # TSNE plot
    mapper_tsne = TSNE(n_components=2, verbose=1, random_state=13200)
    z = mapper_tsne.fit_transform(x) 
    dataframe = pandas.DataFrame()
    dataframe["y"] = y
    dataframe["comp-1"] = z[:,0]
    dataframe["comp-2"] = z[:,1]
    seaborn.scatterplot(x="comp-1", y="comp-2", hue=dataframe.y.tolist(), palette=seaborn.color_palette("hls", 10), data=dataframe).set(title="Cifar10 T-SNE projection") 
    plt.savefig(plot_tsne_path.format(appendix_name), dpi=300)

    # UMAP plot
    mapper_umap = umap.UMAP().fit(x)
    umap.plot.points(mapper_umap, labels=y)
    plt.savefig(plot_umap_path.format(appendix_name), dpi=300)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}...".format(device))
    plot(device)

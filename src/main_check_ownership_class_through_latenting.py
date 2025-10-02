import sys
import numpy
import torch
import src.support.support as support

from sklearn.neighbors import NearestNeighbors
from torch import optim
from src.al_dataset.fashion_mnist_al_dataset import FashionMNISTALDataset
from src.neural_networks.fashion_mnist import fashion_mnist_vae
from src.neural_networks.mnist import mnist_vae
from src.al_dataset.mnist_al_dataset import MNISTALDataset
from src.neural_networks.cnn import Cnn
from src.support.support import clprint, Reason, get_time_in_millis


if __name__ == "__main__":
    support.warm_up()
    support.train_batch_size = 1

    percentage_labeled = 0.001
    model_training_epochs = 10
    n_neighbors = 5
    dataset_name = "mnist"  # "fmnist"


    #############################################################################################################


    if dataset_name == "mnist":
        vae = mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)
        dataset = MNISTALDataset(percentage_labeled)

    elif dataset_name == "fmnist":
        vae = fashion_mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)
        dataset = FashionMNISTALDataset(percentage_labeled)

    else:
        clprint("Unknown dataset...", Reason.WARNING)
        sys.exit(0)

    clprint("Considering {}% ({} samples) of entire dataset to execute the test...".format(percentage_labeled * 100, len(dataset)), Reason.INFO_TRAINING, loggable=True)

    # Training knn on the real samples
    x = []
    y = []
    data = dataset.get_train_loader()
    dataiter = iter(data)
    for batch in dataiter:
        x.append(batch[0].cpu().squeeze().detach().numpy().reshape(1, -1))
        y.append(int(batch[1].squeeze().detach().numpy()))

    clprint("Training knn (n_neighbors={})...".format(n_neighbors), Reason.INFO_TRAINING)
    latented_start_time = get_time_in_millis()
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(numpy.squeeze(numpy.array(x)))
    latented_end_time = get_time_in_millis()

    clprint("Evaluating technique's accuracy...", Reason.INFO_TRAINING)
    unlabeled_dict = dataset.test_dict
    n_correct_samples = 0
    for point in unlabeled_dict.keys():
        latent_space_representation = torch.unsqueeze(point, 0).cpu().squeeze().detach().numpy().reshape(1, -1)
        neighbors = knn.kneighbors(latent_space_representation, return_distance=False)
        n_for_each_class = {}
        for neighbor_index in neighbors[0]:
            if y[neighbor_index] in n_for_each_class.keys():
                n_for_each_class[y[neighbor_index]] += 1

            else:
                n_for_each_class[y[neighbor_index]] = 1

        max = -1
        majority_class = None
        for current_class in n_for_each_class.keys():
            if n_for_each_class[current_class] > max:
                max = n_for_each_class[current_class]
                majority_class = current_class

        if majority_class == unlabeled_dict[point]:
            n_correct_samples += 1

    clprint("The accuracy of the technique on the real samples is {}% reached in {} seconds!".format(((n_correct_samples / len(unlabeled_dict.keys())) * 100), ((latented_end_time - latented_start_time) / 1000)), Reason.OUTPUT_TRAINING, loggable=True)

    # Training knn on the samples in the latent space
    x = []
    y = []
    data = dataset.get_train_loader()
    dataiter = iter(data)
    vae.eval()
    for batch in dataiter:
        x.append(vae.encode(batch[0].to(support.device)).cpu().squeeze().detach().numpy())
        y.append(int(batch[1].squeeze().detach().numpy()))

    clprint("Training knn (n_neighbors={})...".format(n_neighbors), Reason.INFO_TRAINING)
    latented_start_time = get_time_in_millis()
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(numpy.array(x))
    latented_end_time = get_time_in_millis()

    clprint("Evaluating technique's accuracy...", Reason.INFO_TRAINING)
    unlabeled_dict = dataset.test_dict
    n_correct_samples = 0
    for point in unlabeled_dict.keys():
        latent_space_representation = vae.encode(torch.unsqueeze(point, 0).to(support.device)).cpu().squeeze().detach().numpy().reshape(1, -1)
        neighbors = knn.kneighbors(latent_space_representation, return_distance=False)
        n_for_each_class = {}
        for neighbor_index in neighbors[0]:
            if y[neighbor_index] in n_for_each_class.keys():
                n_for_each_class[y[neighbor_index]] += 1

            else:
                n_for_each_class[y[neighbor_index]] = 1

        max = -1
        majority_class = None
        for current_class in n_for_each_class.keys():
            if n_for_each_class[current_class] > max:
                max = n_for_each_class[current_class]
                majority_class = current_class

        if majority_class == unlabeled_dict[point]:
            n_correct_samples += 1

    clprint("The accuracy of the technique on the latent space is {}% reached in {} seconds!".format(((n_correct_samples/len(unlabeled_dict.keys())) * 100), ((latented_end_time - latented_start_time) / 1000)), Reason.OUTPUT_TRAINING, loggable=True)

    # Training a neural network on the real samples
    model = Cnn(support.device)
    optimizer = optim.SGD(model.parameters(), lr=support.cnn_learning_rate, momentum=support.cnn_momentum)
    support.train_batch_size = 32

    clprint("Training neural network...", Reason.INFO_TRAINING)
    training_nn_start_time = get_time_in_millis()
    model.fit(model_training_epochs, None, optimizer, dataset.get_train_loader())
    training_nn_end_time = get_time_in_millis()
    _, accuracy, precision, recall, f1 = model.evaluate(None, dataset.test_loader)

    clprint("Nn's accuracy {}%, precision: {}, recall: {}, F1:{} -> reached in {} seconds!".format(accuracy, precision, recall, f1, ((training_nn_end_time - training_nn_start_time) / 1000)), Reason.OUTPUT_TRAINING, loggable=True)

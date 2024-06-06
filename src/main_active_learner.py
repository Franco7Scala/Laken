import sys
import torch
import src.support.support as support

from torch import optim
from active_learner.simple_active_learner import SimpleActiveLearner
from active_learning_technique.bait_al_technique import BaitALTechnique
from active_learning_technique.qbc_al_technique import QueryByCommiteeALTechnique
from active_learning_technique.lcs_al_technique import LCSALTechnique
from active_learning_technique.query_by_committee.decision_tree_classifier import DecisionTreeClassifier
from active_learning_technique.query_by_committee.random_forest_classifier import RandomForestClassifier
from src.al_dataset.fashion_mnist_al_dataset import FashionMNISTALDataset
from src.neural_networks.fashion_mnist import fashion_mnist_vae
from src.neural_networks.nn import Nn
from src.neural_networks.resnet import ResNet
from src.support.support import Reason, clprint
from active_learning_technique.random_al_technique import RandomALTechnique
from src.neural_networks.mnist import mnist_vae
from src.neural_networks.cnn import Cnn
from src.active_learner.laken_active_learner import LakenActiveLearner
from src.al_dataset.mnist_al_dataset import MNISTALDataset


if __name__ == "__main__":
    support.warm_up()

    percentage_labeled = 0.01
    al_epochs = 10
    training_epochs = 10
    n_samples_to_select = 50
    n_samples_for_human = 50
    n_classes = 10
    use_laken = False
    n_neighbors_for_knn = 5
    al_technique = "rnd"     # "rnd" "lcs" "bait" "qbc"
    model_name = "resnet"    # "cnn" "resnet"
    dataset_name = "mnist"   # "mnist" "fmnist"

    #############################################################################################################
    clprint("Loading {} model...".format(model_name), Reason.INFO_TRAINING)

    if model_name == "cnn":
        model = Cnn(support.device)
        optimizer = optim.SGD(model.parameters(), lr=support.cnn_learning_rate, momentum=support.cnn_momentum)
        criterion = None

    elif model_name == "resnet":
        model = ResNet(support.device)
        optimizer = optim.Adam(model.parameters(), lr=support.resnet_learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

    else:
        clprint("Unknown model named: {}!".format(model_name), Reason.WARNING)
        sys.exit(0)

    clprint("Loading VAE model...", Reason.INFO_TRAINING)

    if dataset_name == "mnist":
        vae = mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)

    elif dataset_name == "fmnist":
        vae = fashion_mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)

    else:
        clprint("Unknown dataset!", Reason.WARNING)
        sys.exit(0)

    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Split dataset selected is {}% yet annotated data!".format(percentage_labeled * 100), Reason.INFO_TRAINING, loggable=True)

    if dataset_name == "mnist":
        al_dataset = MNISTALDataset(percentage_labeled)

    elif dataset_name == "fmnist":
        al_dataset = FashionMNISTALDataset(percentage_labeled)

    else:
        clprint("Unknown dataset!", Reason.WARNING)
        sys.exit(0)

    clprint("Considering {}% ({} samples) of entire dataset to execute the test...".format(percentage_labeled * 100, len(al_dataset)), Reason.INFO_TRAINING, loggable=True)

    if al_technique == "rnd":
        al_technique = RandomALTechnique()

    elif al_technique == "lcs":
        al_technique = LCSALTechnique(model)

    elif al_technique == "bait":
        al_technique = BaitALTechnique(model, al_dataset, n_classes)

    elif al_technique == "qbc":
        al_technique = QueryByCommiteeALTechnique([RandomForestClassifier(al_dataset), model, DecisionTreeClassifier(al_dataset)], n_classes)

    else:
        clprint("Unknown Al technique...", Reason.WARNING)
        sys.exit(0)

    clprint("AL technique selected is {}!".format(al_technique.__class__.__name__), Reason.INFO_TRAINING, loggable=True)

    clprint("Starting AL process...", Reason.INFO_TRAINING)
    if use_laken:
        active_learner = LakenActiveLearner(vae, al_dataset, al_technique, n_samples_for_human, n_neighbors_for_knn)

    else:
        active_learner = SimpleActiveLearner(al_dataset, al_technique)

    active_learner.elaborate(model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer)

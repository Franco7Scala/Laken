import sys
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
from src.neural_networks.fashion_mnist.fashion_mnist_nn import Fashion_MNIST_nn
from src.support.support import Reason, clprint
from active_learning_technique.random_al_technique import RandomALTechnique
from src.neural_networks.mnist import mnist_vae
from src.neural_networks.mnist.mnist_nn import MNIST_nn
from src.active_learner.latented_active_learner import LatentedActiveLearner
from src.al_dataset.mnist_al_dataset import MNISTALDataset


if __name__ == "__main__":
    support.warm_up()

    percentage_labeled = 0.01
    al_epochs = 10
    training_epochs = 10
    n_samples_to_select = 500
    n_samples_for_human = 50
    criterion = None
    n_classes = 10
    al_technique = "rnd"    # "lcs" "bait" "qbc"
    use_latented_al = True
    dataset_name = "mnist"  # "fmnist"

    #############################################################################################################
    clprint("Loading models...", Reason.INFO_TRAINING)
    if dataset_name == "mnist":
        vae = mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)
        model = MNIST_nn(support.device)
        optimizer = optim.SGD(model.parameters(), lr=support.model_learning_rate, momentum=support.model_momentum)

    elif dataset_name == "fmnist":
        vae = fashion_mnist_vae.load_model(support.vae_dim_code, support.model_path, support.device)
        model = Fashion_MNIST_nn(support.device)
        optimizer = optim.SGD(model.parameters(), lr=support.model_learning_rate, momentum=support.model_momentum)

    else:
        clprint("Unknown dataset...", Reason.WARNING)
        sys.exit(0)

    clprint("Loading dataset...", Reason.INFO_TRAINING)
    clprint("Split dataset selected is {}% yet annotated data!".format(percentage_labeled * 100), Reason.INFO_TRAINING, loggable=True)

    if dataset_name == "mnist":
        al_dataset = MNISTALDataset(percentage_labeled)

    elif dataset_name == "fmnist":
        al_dataset = FashionMNISTALDataset(percentage_labeled)

    else:
        clprint("Unknown dataset...", Reason.WARNING)
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
    if use_latented_al:
        active_learner = LatentedActiveLearner(vae, al_dataset, al_technique, n_samples_for_human)

    else:
        active_learner = SimpleActiveLearner(al_dataset, al_technique)

    active_learner.elaborate(model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer)

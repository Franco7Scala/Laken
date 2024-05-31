import os
import torch
import numpy
import time
import warnings
import random

from enum import Enum


warnings.filterwarnings("ignore")


# reproducibility
seed = 0
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms = True
torch.backends.cudnn.benchmark = False

# training parameters
vae_batch_size = 128
vae_dim_code = 4
vae_learning_rate = 1e-3
vae_training_epochs = 100

model_batch_size = 128
model_learning_rate = 1e-3
model_momentum = 0.5
input_shape = (28, 28)

# fixed setup code
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
model_name = "mnist_vae"
dataset_path = "../dataset/{}/"
results_path = "../results"
model_path = "{}/{}.nn".format(results_path, model_name)
log_path = "{}/log_{}.txt"
reconstruction_image_path = "{}/reconstruction_images_{}.pdf".format(results_path, model_name)
latent_space_distribution_path = "{}/latent_space_distribution_{}.pdf".format(results_path, model_name)

# support variables
_time = 0
_called = False


class Reason(Enum):
    INFO_TRAINING = 2
    SETUP_TRAINING = 3
    LIGHT_INFO_TRAINING = 4
    WARNING = 5
    OUTPUT_TRAINING = 6
    OTHER = 7
    NONE = 8


def get_time_in_millis():
    return int(round(time.time() * 1000))


def clprint(text, reason=Reason.NONE, loggable=False):
    global _called
    global _time
    if not _called:
        _called = True
        _time = get_time_in_millis()

    if reason == Reason.INFO_TRAINING:
        code_color = "\033[94m"

    elif reason == Reason.SETUP_TRAINING:
        code_color = "\033[32m"

    elif reason == Reason.LIGHT_INFO_TRAINING:
        code_color = "\033[92m"

    elif reason == Reason.WARNING:
        code_color = "\033[91m"

    elif reason == Reason.OUTPUT_TRAINING:
        code_color = "\033[95m"

    elif reason == Reason.OTHER:
        code_color = "\033[95m"

    else:
        code_color = "\033[0m"

    if loggable:
        path = log_path.format(results_path, _time)
        file = open(path, "a+")
        file.write(text + "\n")
        file.close()

    print(code_color + str(text) + "\033[0m")


def warm_up():
    clprint("Running on {}...".format(device), Reason.SETUP_TRAINING)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

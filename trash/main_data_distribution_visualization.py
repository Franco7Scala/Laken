import torch
import os
import random
import numpy
import matplotlib.pyplot as plt
import auto_encoder
import argparse

from configuration import Configuration
from cifar10_dataset import Cifar10Dataset

base_path = "/root/workspace/DataAugumentation-VAE"
output_path = base_path + "/results"
model_path = output_path + "/model.pth"
plot_umap_path = output_path + "/umap_plot_{}.png"
plot_tsne_path = output_path + "/tsne_plot_{}.png"
dataset_path = base_path + "/data"

size_embedding = 8


def get_configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', nargs='?', default=Configuration.default_batch_size, type=int,
                        help='The size of the batch during training')
    parser.add_argument('--num_training_updates', nargs='?', default=Configuration.default_num_training_updates,
                        type=int, help='The number of updates during training')
    parser.add_argument('--num_hiddens', nargs='?', default=Configuration.default_num_hiddens, type=int,
                        help='The number of hidden neurons in each layer')
    parser.add_argument('--num_residual_hiddens', nargs='?', default=Configuration.default_num_residual_hiddens,
                        type=int, help='The number of hidden neurons in each layer within a residual block')
    parser.add_argument('--num_residual_layers', nargs='?', default=Configuration.default_num_residual_layers, type=int,
                        help='The number of residual layers in a residual stack')
    parser.add_argument('--embedding_dim', nargs='?', default=Configuration.default_embedding_dim, type=int,
                        help='Representing the dimensionality of the tensors in the quantized space')
    parser.add_argument('--num_embeddings', nargs='?', default=Configuration.default_num_embeddings, type=int,
                        help='The number of vectors in the quantized space')
    parser.add_argument('--commitment_cost', nargs='?', default=Configuration.default_commitment_cost, type=float,
                        help='Controls the weighting of the loss terms')
    parser.add_argument('--decay', nargs='?', default=Configuration.default_decay, type=float,
                        help='Decay for the moving averages (set to 0.0 to not use EMA)')
    parser.add_argument('--learning_rate', nargs='?', default=Configuration.default_learning_rate, type=float,
                        help='The learning rate of the optimizer during training updates')
    parser.add_argument('--use_kaiming_normal', nargs='?', default=Configuration.default_use_kaiming_normal, type=bool,
                        help='Use the weight normalization proposed in [He, K et al., 2015]')
    parser.add_argument('--unshuffle_dataset', default=not Configuration.default_shuffle_dataset, action='store_true',
                        help='Do not shuffle the dataset before training')
    parser.add_argument('--data_path', nargs='?', default='data', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str,
                        help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str,
                        help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    parser.add_argument('--original_images_name', nargs='?', default='original_images.png', type=str,
                        help='The file name of the original images used in evaluation')
    parser.add_argument('--validation_images_name', nargs='?', default='validation_images.png', type=str,
                        help='The file name of the reconstructed images used in evaluation')
    args = parser.parse_args()

    return Configuration.build_from_args(args)


def draw_image(image):
    image = image / 2 + 0.5
    npimg = image.squeeze(0).detach().numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
    plt.show()


def generate_string_sequence(length):
    result = ""
    for _ in range(length):
        result += "{} ".format(random.uniform(0, 1))

    return result


def supply_image(config):
    dataset = Cifar10Dataset(config.batch_size, dataset_path, config.shuffle_dataset)
    data = dataset.training_loader
    dataiter = iter(data)
    imgs, _ = next(dataiter)
    draw_image(imgs[0])
    return imgs[0]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    configuration = get_configuration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on {}...".format(device))
    model = auto_encoder.load_model(model_path, configuration, device)
    model = model.to(device)
    sequence_length = configuration.default_num_hiddens * size_embedding * size_embedding
    while True:
        user_input = input("Please put here:\n\t- The data sequence coordinates (numbers divided from space, n. {} elements);\n\t- 'r' to get a random point in the latent space;\n\t- 'g' to select a well known image;\n\t- 'x' to close.".format(sequence_length))
        if user_input == "x":
            print("Bye, see you water!")
            break

        else:
            if user_input == "g":
                input_data = supply_image(configuration)
                tensor = model.compress(input_data.unsqueeze(0).to(device))
                to_print = ""
                for value in tensor.reshape(-1):
                    to_print += "{} ".format(value.item())

                print(to_print)

            else:
                if user_input == "r":
                    user_input = generate_string_sequence(sequence_length)

                tokens = user_input.split(" ")
                tensor = torch.zeros([sequence_length])
                i = 0
                for token in tokens:
                    if token != "":
                        tensor[i] = float(token)
                        i += 1

                tensor = tensor.reshape((1, configuration.embedding_dim, size_embedding, size_embedding)).to(device)

            image = model.reconstruct(tensor)
            draw_image(image.to("cpu"))

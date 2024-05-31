import numpy
import torch
import random
import src.support.support as support

from src.al_dataset.dataset import Dataset
from torch.utils.data import DataLoader


class AbstractALDataset:

    def __init__(self, percentage_labeled, test_dataset, train_dataset, shape_data):
        self.percentage_labeled = percentage_labeled
        self.shape_data = shape_data
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=support.model_batch_size, shuffle=False)
        self.x_labeled = []
        self.y_labeled = []
        self.unlabeled_dict = {}
        self.test_dict = {}
        x_dataset = []
        y_dataset = []
        for _, data in enumerate(train_dataset):
            image, label = data
            x_dataset.append(image)
            y_dataset.append(label)

        map_training_dataset = list(zip(x_dataset, y_dataset))
        random.shuffle(map_training_dataset)
        x_dataset, y_dataset = zip(*map_training_dataset)
        for i in range(len(train_dataset)):
            if i < len(train_dataset) * percentage_labeled:
                self.x_labeled.append(x_dataset[i])
                self.y_labeled.append(y_dataset[i])

            else:
                self.unlabeled_dict[x_dataset[i]] = y_dataset[i]

        self.oracle_x_labeled = self.x_labeled.copy()
        self.oracle_y_labeled = self.y_labeled.copy()
        size = 1
        for dim in self.shape_data:
            size *= dim

        self.test_x = numpy.zeros((len(test_dataset), size))
        self.test_y = numpy.zeros(len(test_dataset))
        i = 0
        for _, data in enumerate(test_dataset):
            image, label = data
            self.test_dict[image] = label
            self.test_x[i] = image.cpu().detach().numpy().reshape(1, -1)
            self.test_y[i] = label
            i += 1

    def __len__(self):
        return len(self.x_labeled)

    def get_unlabeled_data(self):
        return list(self.unlabeled_dict.keys())

    def annotate(self, x_to_label):
        for key in x_to_label:
            current_x = key
            current_y = self.unlabeled_dict.pop(key)
            self.x_labeled.append(current_x)
            self.y_labeled.append(current_y)
            self.oracle_x_labeled.append(current_x)
            self.oracle_y_labeled.append(current_y)

    def supply_annotation(self, xs, ys):
        for i in range(len(xs)):
            self.x_labeled.append(xs[i])
            self.y_labeled.append(ys[i])
            self.unlabeled_dict.pop(xs[i])

    def get_all_data_loader(self):
        return DataLoader(Dataset(self.shape_data, self.x_labeled.extend(self.unlabeled_dict.keys()), None), batch_size=support.model_batch_size)

    def get_train_loader(self):
        return DataLoader(Dataset(self.shape_data, self.x_labeled, self.y_labeled), batch_size=support.model_batch_size)

    def get_oracle_train_loader(self):
        return DataLoader(Dataset(self.shape_data, self.oracle_x_labeled, self.oracle_y_labeled), batch_size=support.model_batch_size)

    def get_test_loader(self):
        return self.test_loader

    def get_train_numpy(self):
        size = 1
        for dim in self.shape_data:
            size *= dim

        x = numpy.zeros((len(self.x_labeled), size))
        y = numpy.zeros(len(self.x_labeled))
        for i in range(len(self.x_labeled)):
            x[i] = self.x_labeled[i].cpu().detach().numpy().reshape(1, -1)
            y[i] = self.y_labeled[i]

        return x, y

    def get_test_numpy(self):
        return self.test_x, self.test_y

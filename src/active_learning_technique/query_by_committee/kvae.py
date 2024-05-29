import numpy
import torch
import src.support.support as support

from sklearn.neighbors import NearestNeighbors


class Kvae:

    def __init__(self, vae, dataset, n_classes):
        self.n_classes = n_classes
        self.vae = vae.to(support.device)
        self._train_knn(dataset)

    def predict(self, x):
        neighbors = self.knn.kneighbors(self.vae.encode(torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0).to(support.device)).cpu().squeeze().detach().numpy().reshape(1, -1), return_distance=False)
        n_for_each_class = {}
        for neighbor_index in neighbors[0]:
            if self.y[neighbor_index] in n_for_each_class.keys():
                n_for_each_class[self.y[neighbor_index]] += 1
            else:
                n_for_each_class[self.y[neighbor_index]] = 1

        max = -1
        majority_class = None
        for current_class in n_for_each_class.keys():
            if n_for_each_class[current_class] > max:
                max = n_for_each_class[current_class]
                majority_class = current_class

        return numpy.eye(self.n_classes)[int(majority_class)]

    def update(self, dataset):
        self._train_knn(dataset)

    def _train_knn(self, dataset):
        data = dataset.get_train_loader()
        dataiter = iter(data)
        self.x = numpy.zeros((len(data) * support.vae_batch_size, support.vae_dim_code))
        self.y = numpy.zeros(len(data) * support.vae_batch_size)
        index = 0
        for batch in dataiter:
            latented_x = self.vae.encode(batch[0].to(support.device)).cpu().squeeze().detach().numpy()
            latented_y = batch[1].squeeze().detach().numpy()
            for i in range(batch[0].shape[0]):
                self.x[index] = latented_x[i]
                self.y[index] = latented_y[i]
                index += 1

        self.knn = NearestNeighbors(n_neighbors=10)
        self.knn.fit(self.x)

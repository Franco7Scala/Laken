import numpy
import torch
import src.support as support

from sklearn.neighbors import NearestNeighbors
from src.active_learner.simple_active_learner import SimpleActiveLearner
from src.support import clprint, Reason


class LatentedActiveLearner(SimpleActiveLearner):

    def __init__(self, vae, dataset, al_technique, n_samples_to_keep):
        super(LatentedActiveLearner, self).__init__(dataset, al_technique)
        self.n_samples_to_keep = n_samples_to_keep
        self.vae = vae.to(support.device)
        clprint("Keeping all labeled data from dataset to train knn...", Reason.INFO_TRAINING)
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

        clprint("Training knn...", Reason.INFO_TRAINING)
        self.knn = NearestNeighbors(n_neighbors=10)
        self.knn.fit(self.x)

    def elaborate(self, model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer):
        clprint("Selecting {} samples for each AL epoch, {} for human annotation and {} for auto annotation".format(n_samples_to_select, self.n_samples_to_keep, n_samples_to_select - self.n_samples_to_keep), Reason.SETUP_TRAINING, loggable=True)
        super().elaborate(model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer)

    def _select_next_samples(self, n_samples_to_select):
        xs = self.al_technique.select_samples(self.dataset.get_unlabeled_data(), n_samples_to_select)
        clprint("Annotating {} samples...".format(self.n_samples_to_keep), Reason.INFO_TRAINING)
        self.dataset.annotate(xs[:self.n_samples_to_keep])
        clprint("Auto annotating {} samples...".format(self.n_samples_to_select - self.n_samples_to_keep), Reason.INFO_TRAINING)
        self.dataset.supply_annotation(xs[self.n_samples_to_keep:], self.latented_annotation(xs[self.n_samples_to_keep:]))

    def latented_annotation(self, xs):
        ys = []
        for point in xs:
            neighbors = self.knn.kneighbors(self.vae.encode(torch.unsqueeze(point, dim=0).to(support.device)).cpu().squeeze().detach().numpy().reshape(1, -1), return_distance=False)
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

            ys.append(int(majority_class))

        return ys

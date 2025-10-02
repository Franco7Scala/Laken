import numpy
import torch
import src.support.support as support

from sklearn.cluster import KMeans
from src.active_learner.simple_active_learner import SimpleActiveLearner
from src.support.support import clprint, Reason


class SSActiveLearner(SimpleActiveLearner):

    def __init__(self, neural_model, optimizer, criterion, dataset, al_technique, n_samples_to_keep):
        super(SSActiveLearner, self).__init__(dataset, al_technique)
        self.n_samples_to_keep = n_samples_to_keep
        self.neural_model = neural_model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_k_means()

    def train_k_means(self):
        clprint("Keeping all labeled data from dataset to train k-menas...", Reason.INFO_TRAINING)
        #test_features_cpu = numpy.load("/home/scala/projects/Laken/src/_other/sodeep/weights/best_model.pth.tar", allow_pickle=True)
        test_features_cpu = numpy.random.random((200, 10))
        self.kmeans = KMeans(n_clusters=10, random_state=0).fit(test_features_cpu)
        confusion_matrix = numpy.zeros((10, 10), dtype=numpy.int32)
        for cluster_label, true_label in zip(self.kmeans.labels_, self.dataset.test_dataset.targets):
            confusion_matrix[true_label, cluster_label] += 1

        self.cluster_to_class = {}
        for j in range(10):
            true_label = numpy.argmax(confusion_matrix[:, j])
            self.cluster_to_class[j] = true_label

    def _pseudo_labeling_and_train(self):
        step, T1, T2, alpha = 100, 100, 7500, 3
        threshold = 0.95
        unlabeled_loss_sum = 0
        for unlb_batch_idx, (data, target, _) in enumerate(self.dataset.get_unlabeled_data_loader()):
            self.optimizer.zero_grad()
            # Forward pass for unlabeled data
            self.neural_model.eval()
            with torch.no_grad():
                outputs_unlabeled, _ = self.neural_model.detailed_forward(data.to(self.neural_model.device), batched=True)
                clustering = self.kmeans.predict(outputs_unlabeled.cpu().to(float).numpy())
                kmeans_mapping = [self.cluster_to_class[cluster_label] for cluster_label in clustering]
                _, pseudo_labels = torch.max(outputs_unlabeled, 1)
                probs = torch.softmax(outputs_unlabeled, dim=1)
                max_probs, _ = torch.max(probs, 1)
                mask_cluster = (torch.tensor(kmeans_mapping).to(self.neural_model.device) == pseudo_labels)
                mask_pseudo = max_probs.ge(threshold)
                mask = mask_pseudo & mask_cluster

            self.neural_model.train()
            outputs_unlabeled = self.neural_model(data.to(self.neural_model.device))
            loss_unlabeled = self.criterion(outputs_unlabeled, pseudo_labels)
            loss_unlabeled = loss_unlabeled.mean()
            loss_unlabeled = loss_unlabeled * mask
            loss_unlabeled = alpha_weight(step, T1, T2, alpha) * loss_unlabeled
            loss_unlabeled = loss_unlabeled.mean()
            unlabeled_loss_sum += loss_unlabeled.detach().cpu().item()
            loss_unlabeled.backward()
            self.optimizer.step()
            # For every 50 batches train one epoch on labeled data
            if unlb_batch_idx % 50 == 0:
                self.neural_model.fit(1, self.criterion, self.optimizer, self.dataset.get_train_loader())

    def elaborate(self, model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer):
        clprint("Selecting {} samples for each AL epoch, {} for human annotation and {} for auto annotation".format(n_samples_to_select, self.n_samples_to_keep, n_samples_to_select - self.n_samples_to_keep), Reason.SETUP_TRAINING, loggable=True)
        super().elaborate(model, al_epochs, training_epochs, n_samples_to_select, criterion, optimizer)

    def _select_next_samples(self, n_samples_to_select):
        xs = self.al_technique.select_samples(self.dataset.get_unlabeled_data(), n_samples_to_select)
        clprint("Annotating {} samples...".format(self.n_samples_to_keep), Reason.INFO_TRAINING)
        self.dataset.annotate(xs[:self.n_samples_to_keep])
        clprint("Auto annotating and training {} samples...".format(self.n_samples_to_select - self.n_samples_to_keep), Reason.INFO_TRAINING)
        self._pseudo_labeling_and_train()


def alpha_weight(epoch, T1, T2, af):
    if epoch < T1:
        return 0.0
    elif epoch > T2:
        return af
    else:
        return ((epoch - T1) / (T2 - T1)) * af

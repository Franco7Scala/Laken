import numpy
import torch
import torch.nn.functional as functional
import src.support.support as support

from torch.utils.data import DataLoader
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique
from src.al_dataset.dataset import Dataset
from copy import deepcopy


class BaitALTechnique(AbstractALTechnique):

    def __init__(self, neural_network, dataset, n_classes):
        self.neural_network = neural_network
        self.dataset = dataset
        self.n_classes = n_classes

    def select_samples(self, unlabeled_samples, n_samples_to_select):
        X_already_labeled, _ = self.dataset.get_train_numpy()
        X_unlabeled_samples = torch.stack(unlabeled_samples)
        X_unlabeled_samples = X_unlabeled_samples.reshape((X_unlabeled_samples.shape[0], X_already_labeled.shape[1])).numpy()
        X = numpy.append(X_already_labeled, X_unlabeled_samples, axis=0)
        xt = self._get_exp_grad_embedding(X)
        batch_size = 1000
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        for i in range(int(numpy.ceil(len(X) / batch_size))):
            xt_ = xt[i * batch_size: (i + 1) * batch_size].to(support.device)
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op

        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt
        for i in range(int(numpy.ceil(len(xt2) / batch_size))):
            xt_ = xt2[i * batch_size: (i + 1) * batch_size].to(support.device)
            op = torch.sum(torch.matmul(xt_.transpose(1, 2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op

        selected_indexes = self._select(xt[X_already_labeled.shape[0]:], n_samples_to_select, fisher, init)
        result = []
        for i in selected_indexes:
            result.append(unlabeled_samples[i])

        return result

    def _get_exp_grad_embedding(self, xs):
        emb_dim = self.neural_network.get_embedding_dim()
        self.neural_network.eval()
        n_labels = self.n_classes
        embedding = numpy.zeros([len(xs), n_labels, emb_dim * n_labels])
        for ind in range(n_labels):
            loader_te = DataLoader(Dataset(support.input_shape, xs, xs))
            with torch.no_grad():
                for x, _, index in loader_te:
                    x = x.to(support.device)
                    c_out, out = self.neural_network.detailed_forward(x)
                    out = out.data.cpu().numpy()
                    batch_probs = functional.softmax(c_out, dim=1).data.cpu().numpy()
                    for j in range(len(x)):
                        for c in range(n_labels):
                            if c == ind:
                                embedding[index[j]][ind][emb_dim * c: emb_dim * (c + 1)] = deepcopy(out[j]) * (1 - batch_probs[j][c])

                            else:
                                embedding[index[j]][ind][emb_dim * c: emb_dim * (c + 1)] = deepcopy(out[j]) * (-1 * batch_probs[j][c])

                        embedding[index[j]][ind] = embedding[index[j]][ind] * numpy.sqrt(batch_probs[j][ind])

        return torch.Tensor(embedding)

    def _select(self, X, K, fisher, iterates, lamb=1, n_labeled=0):
        inds_all = []
        dim = X.shape[-1]
        rank = X.shape[-2]

        current_inv = torch.inverse(lamb * torch.eye(dim).to(support.device) + iterates.to(support.device) * n_labeled / (n_labeled + K))
        X = X * numpy.sqrt(K / (n_labeled + K))
        fisher = fisher.to(support.device)

        over_sample = 2
        for i in range(int(over_sample * K)):
            xt_ = X.to(support.device)
            inner_inv = torch.inverse(torch.eye(rank).to(support.device) + xt_ @ current_inv @ xt_.transpose(1, 2)).detach()
            inner_inv[torch.where(torch.isinf(inner_inv))] = torch.sign(inner_inv[torch.where(torch.isinf(inner_inv))]) * numpy.finfo('float32').max
            trace_est = torch.diagonal(xt_ @ current_inv @ fisher @ current_inv @ xt_.transpose(1, 2) @ inner_inv, dim1=-2, dim2=-1).sum(-1)

            # get the smallest unselected item
            trace_est = trace_est.detach().cpu().numpy()
            for j in numpy.argsort(trace_est)[::-1]:
                if j not in inds_all:
                    ind = j
                    break

            inds_all.append(ind)
            xt_ = X[ind].unsqueeze(0).to(support.device)
            inner_inv = torch.inverse(torch.eye(rank).to(support.device) + xt_ @ current_inv @ xt_.transpose(1, 2)).detach()
            current_inv = (current_inv - current_inv @ xt_.transpose(1, 2) @ inner_inv @ xt_ @ current_inv).detach()[0]

        # backward pruning
        for i in range(len(inds_all) - K):
            # select index for removal
            xt_ = X[inds_all].to(support.device)
            inner_inv = torch.inverse(-1 * torch.eye(rank).to(support.device) + xt_ @ current_inv @ xt_.transpose(1, 2)).detach()
            trace_est = torch.diagonal(xt_ @ current_inv @ fisher @ current_inv @ xt_.transpose(1, 2) @ inner_inv, dim1=-2, dim2=-1).sum(-1)
            del_ind = torch.argmin(-1 * trace_est).item()

            # low-rank update (woodbury identity)
            xt_ = X[inds_all[del_ind]].unsqueeze(0).to(support.device)
            inner_inv = torch.inverse(-1 * torch.eye(rank).to(support.device) + xt_ @ current_inv @ xt_.transpose(1, 2)).detach()
            current_inv = (current_inv - current_inv @ xt_.transpose(1, 2) @ inner_inv @ xt_ @ current_inv).detach()[0]
            del inds_all[del_ind]

        return inds_all

import numpy

from scipy.spatial.distance import pdist
from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class QueryByCommiteeALTechnique(AbstractALTechnique):

    def __init__(self, models, n_classes):
        self.models = models
        self.n_classes = n_classes

    def select_samples(self, x, n_samples_to_select):
        disagreements = numpy.zeros(len(x))
        for sample_index in range(len(x)):
            predictions = numpy.zeros((len(self.models), self.n_classes))
            for model_index in range(len(self.models)):
                predictions[model_index] = self.models[model_index].predict(x[sample_index].squeeze())

            # calculating disagreement
            disagreements[sample_index] = numpy.mean(pdist(predictions))

        # selecting more disagreeded samples
        result = []
        best_index = (-disagreements).argsort()[:n_samples_to_select]
        for best_sample_index in best_index:
            result.append(x[best_sample_index])

        return result

    def update(self, dataset):
        for model in self.models:
            model.update(dataset)

import numpy
import src.support as support

from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class LCSALTechnique(AbstractALTechnique):

    def __init__(self, neural_network):
        self.neural_network = neural_network

    def select_samples(self, x, n_samples_to_select):
        selected_samples = []
        confidences = []
        for sample in x:
            # calculating least confidence
            out_model = self.neural_network(sample.to(support.device))[0]
            simple_least_confidence = numpy.nanmax(out_model.cpu().detach().numpy())
            normalized_least_confidence = (1 - simple_least_confidence) * (len(out_model) / (len(out_model) - 1))
            # appending
            if len(selected_samples) >= n_samples_to_select:
                min_confidence = min(confidences)
                min_confidence_index = confidences.index(min_confidence)
                if normalized_least_confidence > min_confidence:
                    selected_samples.pop(min_confidence_index)
                    confidences.pop(min_confidence_index)
                    selected_samples.append(sample)
                    confidences.append(normalized_least_confidence)

            else:
                selected_samples.append(sample)
                confidences.append(normalized_least_confidence)

        return selected_samples

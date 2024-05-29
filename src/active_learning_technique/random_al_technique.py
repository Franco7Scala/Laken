from src.active_learning_technique.abstract_al_technique import AbstractALTechnique


class RandomALTechnique(AbstractALTechnique):

    def select_samples(self, x, n_samples_to_select):
        return x[:n_samples_to_select]
